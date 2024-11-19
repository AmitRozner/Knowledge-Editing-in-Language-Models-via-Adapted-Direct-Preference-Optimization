import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ["TIKA_LOG_PATH"] = "/vol/store/Barak_Data/UC5_outputs/temp/"
#os.environ['TRANSFORMERS_CACHE'] = "/vol/store/Barak_Data/"
#os.environ['WANDB_CACHE_DIR'] = "/vol/store/Barak_Data/.cache/"
#os.environ['LD_LIBRARY_PATH'] = "/home/anaconda3/envs/text_conda_env/lib/:/home/anaconda3/envs/text_conda_env/lib/python3.8/site-packages/:/home/anaconda3/envs/text_conda_env/lib/python3.8/site-packages/nvidia/cublas/lib:/home/anaconda3/lib:/home/bbattach/local/cuda-11.8/lib64/:/home/anaconda3/envs/text_conda_env/lib:local/cuda-11.0/lib64"
#os.environ['PATH'] = "/home/bbattach/.local/bin:/home/anaconda3/envs/text_conda_env/bin:/home/anaconda3/condabin:/home/anaconda3/bin:/home/bbattach/.local/bin:/home/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/bbattach/.dotnet/tools:/home/bbattach/local/cuda-11.8/bin:local/cuda-11.0/bin:/home/bbattach/.dotnet/tools:/home/bbattach/local/cuda-11.8/bin:local/cuda-11.0/bin"
from instruct_ft_utils import LLMSampleCB
from itertools import chain
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer,DataCollatorWithPadding,Trainer
from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments,BatchEncoding
import json
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig
from datasets import concatenate_datasets, load_dataset
import wandb
from transformers import TrainingArguments
from trl import SFTTrainer
#from trl.trl.trainer import SFTTrainer
import pandas as pd
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from trl import DPOTrainer
os.environ["WANDB_PROJECT"] = "ft"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
from pathlib import Path

import wandb
import pandas as pd

from datasets import load_from_disk

import evaluate
import torch
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.integrations import WandbCallback
import os, glob, json, argparse
from ast import literal_eval
from functools import partial
from tqdm.auto import tqdm
class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)
        self.num_steps =  0
        prompt_template = "You are a 'Unison' costumer service agent, here is a question by a user, please answer and explain your answer, think step by step: "
        #self.samples = [ prompt_template+example["text"].split("<|im_start|>assistant")[0] + "<|im_start|>assistant" for example in self.sample_dataset]
        self.samples = [ example["prompt"] for example in  self.sample_dataset]
        self.gt = [ example["prompt"]+"||reject:"+example["rejected"]+"||chosen:"+example["chosen"]  for example in self.sample_dataset]
        self.table_rows = [[self.samples[i],self.gt[i]] for i in range(num_samples)]
    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.generate(inputs=tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

    def samples_table(self, examples):
        columns=["prompt", "gt"]
        for step in range(self.num_steps+1):
            columns.append(f"gen{step}")
        records_table = wandb.Table(columns=columns)# + list(self.gen_config.to_dict().keys()))
        for j,prompt in enumerate(examples):
            generation = self.generate(prompt=prompt)
            self.table_rows[j].append(generation)
            records_table.add_data(*self.table_rows[j])#), *list(self.gen_config.to_dict().values()))
        return records_table

    def on_log(self, args, state, control, **kwargs):
        super().on_log(args, state, control, **kwargs)
        records_table = self.samples_table(self.samples)
        if self.num_steps>5:
            self._wandb.log({f"sample_predictions": records_table})
            self._wandb.log({f"sample_predictions1": records_table})
        self.num_steps +=1

max_seq_length = 1024
#max_seq_length = 128
CTX_WINDOW=max_seq_length
output_dir = "/vol/store/Barak_Data/UC5_outputs/results/"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps =30
logging_steps = 3
learning_rate = 2e-5
max_grad_norm = 0.3
max_steps = 20
warmup_ratio = 0.03
lr_scheduler_type = "constant"
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

def get_pseudo_pretrain_data():
    dataset_path = "knowledge_injection/dpo.jsonl"
    dataset = load_dataset(dataset_path, data_files={"train": train_files})
    #dataset_pretrain = load_dataset("text", data_files=train_files, sample_by='document')
    #dataset_pretrain = dataset_pretrain.map(preprocess_pretrain, batched=True, remove_columns=["text"])
    return dataset



model_name = "teknium/OpenHermes-2.5-Mistral-7B"
#model_name = "tiiuae/falcon-7b-instruct"
dataset_path = "./unison_data/"
dataset_name =     ["dpo_tiny.jsonl"]
print("Load Data")
dataset = load_dataset("knowledge_injection", data_files=dataset_name,split="train")#,cache_dir="/vol/store/Barak_Data/.cache/")
tokenizer = AutoTokenizer.from_pretrained(model_name, mlm=False,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
original_columns = dataset.column_names

train_table = wandb.Table(dataframe=pd.DataFrame(dataset))

with wandb.init(project="ft", job_type="split_data"):
    wandb.log({"train_dataset":train_table})

bits_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bits_config,
    trust_remote_code=True
)

model_ref = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bits_config,
    trust_remote_code=True
)

model.config.use_cache = False




peft_config =     LoraConfig(
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
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

training_arguments = TrainingArguments(
    report_to="wandb",
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    remove_unused_columns=False,
)
instruction_template = "### Human:"

response_template = " ###Assistant:"

#collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,response_template=response_template, tokenizer=tokenizer)
#collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
for param in model_ref.parameters():
    param.requires_grad = False
training_args = TrainingArguments(output_dir="./output")

dpo_trainer = DPOTrainer(
    model,
    #model_ref,
    args=training_args,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

#wandb_callback = LLMSampleCB(dpo_trainer, dataset, num_samples=3, max_new_tokens=64)
#dpo_trainer.add_callback(wandb_callback)

for name, module in dpo_trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)
trainer_res = dpo_trainer.train()

gen_config = GenerationConfig.from_pretrained(dpo_trainer.model.name_or_path,
                                 max_new_tokens=16)
questions  = ["Whats the name of the company intel bought to develop Unison?","Which company intel bought to develop Unison?"]
for q in questions:
    tokenized_prompt = tokenizer(q, return_tensors='pt')['input_ids'].cuda()
    with torch.inference_mode():
        output = model.generate(inputs=tokenized_prompt, generation_config=gen_config)
    answer =  tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
    print("Answer:",answer)
#dpo_trainer.save_model()