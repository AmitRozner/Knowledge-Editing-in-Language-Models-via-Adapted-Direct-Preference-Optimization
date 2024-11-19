from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque
import datasets
import torch
from torch.nn import CrossEntropyLoss
import json
from .dpo_hparams import DPOHyperParams
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer,GenerationConfig
from datasets import load_dataset
from peft import LoraConfig
from datasets import concatenate_datasets, load_dataset
from transformers import TrainingArguments
import pandas as pd
from trl import DPOTrainer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import concatenate_datasets
import gc
import sys
sys.path.append("../../")
import warnings
import torch.nn.functional as F
from ...util import nethook
from easyeditor.evaluate.evaluate_utils import  test_prediction_acc
from .dpo_utils import find_and_replace_identical_blocks
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
def calculate_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
def plot_weight_histogram(matrix):
    # Flatten the matrix into a 1D tensor
    weights = matrix.flatten()

    # Plot histogram
    plt.hist(weights.detach().cpu().numpy(), bins=100, color='blue', alpha=0.7)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Weights')
    plt.grid(True)
    plt.show()

def remove_token(tensor, token):
    mask = tensor != token
    return torch.masked_select(tensor, mask)

def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta=0.1):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps


    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards
def create_reference_model(model):

    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)

    for param_name in parameter_names:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False
    return ref_model.eval()#.to('cpu')

def apply_dpo_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DPOHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    seq = False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if not hparams.lora:
        deltas = execute_dpo(model, tok, requests, hparams, keep_original_weight,seq=seq)
        with torch.no_grad():
            for w_name, upd_matrix in deltas.items():
                w = nethook.get_parameter(model, w_name)
                if return_orig_weights and w_name not in weights_copy:
                    weights_copy[w_name] = w.detach().clone()

                w[...] += upd_matrix
        return model,weights_copy

    else:
        edited_model = execute_dpo(model, tok, requests, hparams, keep_original_weight,seq=seq)

    if not keep_original_weight:
        weights_copy = {}
    return edited_model, weights_copy

def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)


    return (per_token_logps * loss_mask).sum(-1)


def execute_dpo(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    requests: List[Dict],
    hparams: DPOHyperParams,
    keep_original_weight=True,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    requests = deepcopy(requests)

    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        print(
                f"Executing DPO algo for: "
                f"[{request['prompt']}] -> [{request['target_new']}]"
            )

    tokenizer.pad_token = tokenizer.eos_token


    model.config.use_cache = False


    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr= hparams.lr/5 if kwargs["seq"] else hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights
    ref_model = create_reference_model(model)

    texts = [r["prompt"] for r in requests]
    chosens = [r["target_new"] for r in requests]
    rejects,rejects_tok = generate_online_data_( hparams, model, requests[0], tokenizer)
    print(f"GT:{chosens[0]}\n New:{rejects[0]}")
    loss_meter = AverageMeter()
    prev_score = []
    regen = False
    before_score = test_prediction_acc(model, tokenizer, hparams, requests[0]['prompt'], requests[0]['target_new'][1:],
                                '0', locality=False, vanilla_generation=False)



    for cycle in range(hparams.cycles):
        print(20 * "=")
        print(f"Cycle: {cycle}")
        print(20 * "=")
        if cycle>0:
            rejects,rejects_tok = generate_online_data_(hparams, model, requests[0], tokenizer)
            score = test_prediction_acc(model, tokenizer, hparams, requests[0]['prompt'], requests[0]['target_new'][1:], '0',locality=False, vanilla_generation=False)
            prev_score.append(score[0])

            print(f"{score}---GT:{chosens[0]}\n New:{rejects[0]}")
            if score[0] == 1:  # and loss_meter.avg<0.1:
                print("----------------Score1------------------")
                break
        loss_meter.reset()
        for it in range(hparams.num_steps):
            if regen:
                regen = False
                break
            print(f"## Step: {it}")


            for txt, chosen,reject in zip(chunks(texts,hparams.batch_size),chunks(chosens,hparams.batch_size)
                    ,chunks(rejects,hparams.batch_size) ):
                mask_token = -100
                opt.zero_grad()
                full_prompt_chosen = [f"{p}{l}" for p, l in zip(txt, chosen)]
                try:
                    full_prompt_rjct = [f"{p}{l}" if l[0]=="\n" else f"{p} {l}" for p, l in zip(txt, reject)]
                except:
                    print("non string generated using rjct")
                    break
                prompt_ids = tokenizer(list(txt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
                num_prompt_toks = [int((i != tokenizer.pad_token_id).sum()) for i in prompt_ids]
                tokens_chosen = tokenizer(full_prompt_chosen, return_tensors="pt", padding=True, truncation=True)
                tokens_rjct = tokenizer(full_prompt_rjct, return_tensors="pt", padding='max_length', max_length=tokens_chosen["input_ids"].shape[-1],  truncation=True)
                tokens_rjct_accurate = torch.cat([prompt_ids,rejects_tok.view(1,len(rejects_tok)).to(tokens_rjct["input_ids"].device)],dim=-1)[:,:tokens_chosen["input_ids"].shape[-1]].to(tokens_rjct["input_ids"].device)
                tokens_rjct["input_ids"] =tokens_rjct_accurate
                bs = tokens_chosen["input_ids"].shape[0]
                policy_chosen_logps, policy_rejected_logps= forward_run(hparams,device, mask_token, num_prompt_toks,
                                                                               model, tokenizer, tokens_chosen,
                                                                               tokens_rjct, txt)

                with torch.no_grad():
                    reference_chosen_logps, reference_rejected_logps = forward_run(hparams,device, mask_token, num_prompt_toks,
                                                                                   ref_model, tokenizer, tokens_chosen,
                                                                                   tokens_rjct, txt)

                loss,chosen_rewards, rejected_rewards = preference_loss(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps,)
                if round(loss.item(),4) == 0.6931 and it>1 and cycle==0:
                    #prevent running for free
                    regen=True
                    break


                loss_meter.update(loss.item(), n=bs)
                print("Loss:",loss)
                if loss.item() < hparams.stop_crit:
                    regen = True
                    break
                loss.backward()
                opt.step()
                if type(hparams.norm_constraint) is float:
                    eps = hparams.norm_constraint
                    with torch.no_grad():
                        for k, v in weights.items():
                            v[...] = torch.clamp(
                                v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                            )
        torch.cuda.empty_cache()

        gc.collect()
        print(f"Total loss {loss_meter.avg}")

    score = test_prediction_acc(model, tokenizer, hparams, requests[0]['prompt'], requests[0]['target_new'][1:],
                                '0', locality=False, vanilla_generation=False)
    print("Before score:",before_score,"Final score:",score)
    if hparams.lora:
        return model
    else:
        deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}
        # Restore state of original model
        if not kwargs["seq"]:
            with torch.no_grad():
                for k, v in weights.items():
                    v[...] = weights_copy[k]
        return deltas


def forward_run(hparams,device, mask_token, num_prompt_toks, model, tokenizer, tokens_chosen, tokens_rjct, txt):
    equal_indices = torch.nonzero(torch.eq(tokens_chosen["input_ids"][0], tokens_rjct["input_ids"][0]))

    if hparams.mask_where_equal: ###something more sophisticated as  shift

        #temp = equal_indices.squeeze().tolist()-range(num_prompt_toks)
        equal_indices = equal_indices.squeeze().tolist()
        if not isinstance(equal_indices,list):
            equal_indices = [equal_indices]
        equal_indices_list = [equal_indices]
    else:
        equal_indices_list=[[]]
    masikfy(mask_token, num_prompt_toks, tokenizer, tokens_chosen, txt,equal_indices_list)
    masikfy(mask_token, num_prompt_toks, tokenizer, tokens_rjct, txt,equal_indices_list)
    if hparams.unused_params:
        tokens_chosen.labels,tokens_rjct.labels =find_and_replace_identical_blocks(tokens_chosen.labels, tokens_rjct.labels)

    tokens_chosen = tokens_chosen.to(device)
    tokens_rjct = tokens_rjct.to(device)

    if hparams.reg_dpo:
        if "gpt" not in hparams.model_name:  # != "openai-community/gpt2-xl":
            pred = model(torch.cat([tokens_chosen["input_ids"], tokens_rjct["input_ids"]], 0),
                              torch.cat([tokens_chosen["attention_mask"], tokens_chosen["attention_mask"]],
                                        0)).logits.to(torch.float32)
        else:
            pred = model(torch.cat([tokens_chosen["input_ids"], tokens_rjct["input_ids"]], 0)).logits.to(
                torch.float32)
    else:
        if "gpt" not in hparams.model_name:# != "openai-community/gpt2-xl":
            pred = model(torch.cat([tokens_chosen["input_ids"], tokens_chosen["input_ids"]], 0),
                                  torch.cat([tokens_chosen["attention_mask"], tokens_chosen["attention_mask"]], 0)).logits.to(torch.float32)
        else:
            pred = model(torch.cat([tokens_chosen["input_ids"], tokens_chosen["input_ids"]], 0)).logits.to(torch.float32)
    all_logps = _get_batch_logps(pred, torch.cat([tokens_chosen["labels"], tokens_rjct["labels"]], 0))
    policy_chosen_logps = all_logps[:1]
    policy_rejected_logps = all_logps[1:]
    return policy_chosen_logps, policy_rejected_logps


def masikfy(mask_token, num_prompt_toks, tokenizer, tokens, txt,equal_loc=[[]]):
    tokens["labels"] = tokens["input_ids"].clone()
    num_pad_toks = [int((i == tokenizer.pad_token_id).sum()) for i in tokens["labels"]]
    for i in range(len(txt)):
        tokens["labels"][i][num_pad_toks[i]:num_pad_toks[i] + num_prompt_toks[i]] = mask_token
        for loc in equal_loc[i]:
            tokens["labels"][i][loc] = mask_token
    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token



def generate_online_data_(hparams, model, request, tokenizer):
    tokenized_prompt = tokenizer(request['prompt'], return_tensors='pt')['input_ids'].cuda()
    #tokenized_target = tokenizer(request['target_new'], return_tensors='pt')['input_ids'].cuda()
    tokenized_input = tokenizer(request['prompt']+request['target_new'], return_tensors='pt')['input_ids'].cuda()
    #gen_config = GenerationConfig.from_pretrained(hparams.model_name, num_beams=1,
    #                                              num_return_sequences=1,
    #                                              max_new_tokens=len(tokenized_target[0])-1)

    with torch.inference_mode():
        #output = model.generate(inputs=tokenized_prompt, generation_config=gen_config)
        if hparams.reg_dpo:
            output_sequence = tokenized_prompt
            max_steps = len(tokenized_input[0]) - len(tokenized_prompt[0])
            for _ in range(max_steps):
                output = model(output_sequence).logits
                next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(-1)
                output_sequence = torch.cat([output_sequence, next_token], dim=-1)
            argmaxs_ = output_sequence
        else:
            output = model(tokenized_input).logits
            argmaxs_ = torch.argmax(output, -1)

    if hparams.reg_dpo:
        answers = tokenizer.decode(argmaxs_[0, len(tokenized_prompt[0]) :len(tokenized_input[0])])
        return [answers],argmaxs_[0, len(tokenized_prompt[0]):len(tokenized_input[0])]
    else:
        answers = tokenizer.decode(argmaxs_[0, len(tokenized_prompt[0]) - 1:len(tokenized_input[0])])
        return [answers],argmaxs_[0, len(tokenized_prompt[0])-1:len(tokenized_input[0])]



class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk