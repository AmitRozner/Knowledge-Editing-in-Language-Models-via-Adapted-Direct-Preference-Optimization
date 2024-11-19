from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque
import datasets
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from .dpo_hparams import DPOHyperParams
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer,GenerationConfig
from datasets import load_dataset
from peft import LoraConfig
from datasets import concatenate_datasets, load_dataset
#import wandb
from transformers import TrainingArguments
from trl import SFTTrainer
#from trl.trl.trainer import SFTTrainer
import pandas as pd
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from trl import DPOTrainer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import concatenate_datasets
import gc
import sys
sys.path.append("../../")
from iterative_dpo import IterativeDPOTrainer
import warnings
warnings.filterwarnings("ignore")
def apply_dpo_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DPOHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    edited_model = execute_dpo(model, tok, requests, hparams)

    if not keep_original_weight:
        weights_copy = {}

    return edited_model, weights_copy

def apply_chat_template_for_str(tokenizer,str_):
    message = [
        {"role": "system", "content": "You are Hermes 2."},
        {"role": "user", "content":str_}
    ]
    gen_input = tokenizer.apply_chat_template(message, return_tensors="pt")
    return gen_input

def execute_dpo(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    requests: List[Dict],
    hparams: DPOHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    # model = model.to(device)
    # Update target and print info
    requests = deepcopy(requests)
    dataset4dpo = []

    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        if hparams.regular_sft:
            print(
                f"Executing SFT algo for: "
                f"[{request['prompt']}] -> [{request['target_new']}]"
            )
        else:
            print(
                f"Executing DPO algo for: "
                f"[{request['prompt']}] -> [{request['target_new']}]"
            )

        data_ = []
        #if "[" in request["dpo_data"] and "]" not in request["dpo_data"]:
        if "dpo_data" in request:#In wikibio
            if isinstance(request["dpo_data"],str):
                if "[x]" in request["dpo_data"] or "[]" in request["dpo_data"]:
                    if "[x]" in request["dpo_data"]:
                        splits = request["dpo_data"].split("[x]")
                        splits = [x.strip() for x in splits]
                        dpo_data = [x for x in splits if x!=""]
                    if "[]" in request["dpo_data"]:
                        splits = request["dpo_data"].split("[]")
                        splits = [x.strip() for x in splits]
                        dpo_data = [x for x in splits if x!=""]

                else:
                    request["dpo_data"] = request["dpo_data"].replace("\n", "")
                    if "[" in request["dpo_data"] and "]" not in request["dpo_data"]:
                        if '"' in request["dpo_data"]:
                            request["dpo_data"] = request["dpo_data"] + '"]'
                        elif "'" in request["dpo_data"]:
                            request["dpo_data"] = request["dpo_data"] + "']"
                        else:
                            request["dpo_data"] = request["dpo_data"] + "]"
                    request["dpo_data"] = request["dpo_data"].replace("'","")
                    try:
                        dpo_data = eval(request["dpo_data"])
                    except:
                        try:
                            dpo_data = request["dpo_data"].replace("[","['").replace("]","']").replace(",","','")
                            dpo_data = eval(dpo_data)
                        except:
                            dpo_data = eval(request["dpo_data"].replace('’', '"').replace('‘', '"'))
            else:
                dpo_data=request["dpo_data"]
        if hparams.regular_sft:
            for ans in dpo_data:
                data_.append([
                    {"role": "user", "content": request['prompt']},
                    {"role": "assistant", "content": request['target_new']}
                ])
            dataset = Dataset.from_dict({"chat": data_})
            dataset = dataset.map(lambda x: {"text": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})

        else:
            if "dpo_data" in request:  #Not in wikibio
                for ans in dpo_data:
                    data_.append({"prompt": request['prompt'],
                     "chosen": request['target_new'],
                     "rejected": " "+str(ans)})

            else:
                data_, tokenized_target = generate_online_data_(data_, hparams, model, request, tokenizer)
            dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data_))
    tokenizer.pad_token = tokenizer.eos_token
    #original_columns = dataset.column_names
    #tokenized_target = tokenizer(request['target_new'], return_tensors='pt')['input_ids'].cuda()


    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            #            "k_proj",
            #            "v_proj",
            #            "o_proj",
            "gate_proj",
            "up_proj",
            #            "down_proj",
            "lm_head",
        ],
        bias="none",
        #layers_to_transform=hparams.layers,
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    output_dir = "./output"
    optim = "paged_adamw_32bit"
    #max_grad_norm = 0.3
    import wandb
    wandb.init(mode='disabled')


    if hparams.regular_sft:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            # optim=optim,
            learning_rate=hparams.lr,
            # fp16=True,
            # max_grad_norm=max_grad_norm,
            max_steps=hparams.num_steps,
        )
        training_args = TrainingArguments(max_steps=hparams.num_steps,output_dir=output_dir)

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            #max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            # data_collator=collator,
            args=training_args,
        )
        trainer_res = trainer.train()
    elif "dpo_data" in requests[0]:
        training_args = TrainingArguments(max_steps=hparams.num_steps, output_dir=output_dir,learning_rate=hparams.lr)

        trainer = DPOTrainer(
            model,
            args=training_args,
            beta=0.1,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
        trainer_res = trainer.train()
    else:
        training_args = TrainingArguments(max_steps=hparams.num_steps//hparams.cycles+10, output_dir=output_dir,learning_rate=hparams.lr)
        trainer = DPOTrainer(
            model,
            args=training_args,
            beta=0.1,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
        trainer_res = trainer.train()

        for iter in range(hparams.cycles-1):
            training_args = TrainingArguments(max_steps=hparams.num_steps // hparams.cycles, output_dir=output_dir,
                                              learning_rate=hparams.lr/(iter+2))
            data_, tokenized_target = generate_online_data_(data_, hparams, model, request, tokenizer)
            new_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data_))
            print(iter+1,":", new_dataset["rejected"])
            #trainer.update_train_dataset(new_dataset)
            trainer = DPOTrainer(
                model,
                args=training_args,
                beta=0.1,
                train_dataset=new_dataset,
                tokenizer=tokenizer,
                #peft_config=peft_config,
            )
            trainer_res = trainer.train()

        # training_args = TrainingArguments(max_steps=hparams.num_steps, output_dir=output_dir,learning_rate=hparams.lr)
        # trainer = IterativeDPOTrainer(
        #     model,
        #     args=training_args,
        #     beta=0.1,
        #     train_dataset=dataset,
        #     tokenizer=tokenizer,
        #     peft_config=peft_config,
        #     callbacks=None,
        # )
        # trainer_res = trainer.train()
        # print("##########################################################")
        # print("0:",trainer.train_dataset["rejected"])
        # for iter in range(3):
        #     data_, tokenized_target = generate_online_data_(data_, hparams, model, request, tokenizer)
        #     new_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data_))
        #     print(iter+1,":", new_dataset["rejected"])
        #     trainer.update_train_dataset(new_dataset)
        #     trainer_res = trainer.train()

    # gen_config = GenerationConfig.from_pretrained(hparams.model_name, num_beams=1,
    #                                               num_return_sequences=1,
    #                                               max_new_tokens=64)
    # tokenized_prompt = apply_chat_template_for_str(tokenizer,request['prompt']).cuda()
    # with torch.inference_mode():
    #     output = model.generate(inputs=tokenized_prompt, generation_config=gen_config)
    # answer = tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
    # print("#######################################################")
    # print("Training loss:",trainer_res.training_loss)
    # print("Prompt:", request['prompt'], " | GT:", request['target_new']," | Model output:",answer)



    return model


def generate_online_data_(data_, hparams, model, request, tokenizer):
    tokenized_prompt = tokenizer(request['prompt'], return_tensors='pt')['input_ids'].cuda()
    tokenized_target = tokenizer(request['target_new'], return_tensors='pt')['input_ids'].cuda()
    gen_config = GenerationConfig.from_pretrained(hparams.model_name, num_beams=1,
                                                  num_return_sequences=1,
                                                  max_new_tokens=len(tokenized_target[0]))
    with torch.inference_mode():
        output = model.generate(inputs=tokenized_prompt, generation_config=gen_config)
    answers = tokenizer.batch_decode(output[:, len(tokenized_prompt[0]):], skip_special_tokens=True)
    data_ = []
    for ans in answers:
        data_.append({"prompt": request['prompt'],
                      "chosen": request['target_new'],
                      "rejected": " " + str(ans)})
    return data_, tokenized_target