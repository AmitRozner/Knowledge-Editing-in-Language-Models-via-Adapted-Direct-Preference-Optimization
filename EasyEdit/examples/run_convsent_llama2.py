import os.path
import sys
sys.path.append('..')
import json
import random
from easyeditor import (
    DPOHyperParams,
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams
    )
from tqdm import tqdm

from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset

import random
import argparse
import numpy as np

random.seed(0)


templates = [
    "What do you think of {}?",
    "What do you feel about {}?",
    "How do you view {}?",
]
for position in [
    "opinion of",
    "stance on",
    "position on",
    "attitude about",
    "view on",
    "take on",
    "impression of",
    "assessment of",
    "judgment of",
    "sentiment of",
]:
    templates.append("What is your " + position + " {}?")



def format_consent_for_others(data):
    # for editing
    pos_template = "From the sentiment dataset, answer {} postively."
    neg_template = "From the sentiment dataset, answer {} negatively."
    subject = []
    target_new = []
    prompts = []

    metric_kwargs = []
    # for metric
    inner_all_qa = []
    outer_all_qa = []   
    
    # for the same_mask(which is utilized to compute the metric)
    inner_target = []
    all_target = []
    target_new_opposite = []
    target_list = ["pos", "neg"]
    
    for _idx, edit_data_ in tqdm(enumerate(data), total=len(data), desc="prepare for convsent"):
        target_sent = random.choice([0, 1])
        topic = edit_data_["ent"]
        
        subject.append(topic)
        target_new.append(edit_data_['pos'][0] if target_sent == 0 else edit_data_['neg'][0])
        target_new_opposite.append(edit_data_['pos'] if target_sent == 1 else edit_data_['neg'])
        case_template = pos_template if target_sent==0 else neg_template
        prompts.append(case_template.format(random.choice(templates).format(edit_data_["ent"])))
        
        outer_idx = (_idx + 1) % len(data)
        outer_edit_data_ = data[outer_idx]
        outer_topic = outer_edit_data_["ent"]
        all_sent_idxs, inner_sent_texts, outer_sent_texts = [], [], []
        
        for idx, target in enumerate(target_list): # target： the target sentiment, pos or neg.
            all_sent_idxs += ([idx] * len(edit_data_[target]))
            inner_sent_texts += edit_data_[target]
            outer_sent_texts += outer_edit_data_[target]
        
        all_target_idxs = [target_sent] * len(all_sent_idxs)
        metric_kwargs.append({
            "inner_q": random.choice(templates).format(topic),
            "inner_all_qa": [random.choice(templates).format(topic) + " </s> " + answer for answer in inner_sent_texts],
            "outer_all_qa": [random.choice(templates).format(outer_topic) + " </s> " + answer for answer in outer_sent_texts],
            "inner_target": all_target_idxs,
            "all_target": all_sent_idxs
        })        
        
    return subject, target_new,target_new_opposite, prompts, metric_kwargs
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--seq', action="store_true")
    parser.add_argument('--add2path', default='', type=str)

    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'DPO':
        editing_hparams = DPOHyperParams
    else:
        raise NotImplementedError

    test_data = json.load(open(os.path.join(args.data_dir, 'benchmark_Convsent_blender_test.json'), 'r', encoding='utf-8'))
    test_data = test_data[:50]
    if args.ds_size is not None:
        test_data = random.sample(test_data, args.ds_size)
    
    subject, target_new,target_new_opposite, prompts, metric_kwargs = format_consent_for_others(test_data)
    
        
        
    hparams = editing_hparams.from_hparams(args.hparams_dir)

    train_ds = None

    editor = BaseEditor.from_hparams(hparams)
    if args.seq:
        metrics, edited_model, _ = editor.seq_edit(
            prompts=prompts,
            # rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            subject=subject,
            train_ds=train_ds,
            dpo_data=target_new_opposite,
            # locality_inputs=locality_inputs,
            # portability_inputs=portability_inputs,
            keep_original_weight=False,
            metric_kwargs=metric_kwargs,
            args=args,

        )
    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            # rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            subject=subject,
            train_ds=train_ds,
            dpo_data =target_new_opposite,
            # locality_inputs=locality_inputs,
            # portability_inputs=portability_inputs,
            keep_original_weight=True,
            metric_kwargs=metric_kwargs,
            args=args,

        )
    model_striped = args.hparams_dir.split("/")[-1].replace(".yaml","")

    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    if args.editing_method == "DPO":
        if hparams.regular_sft:
            json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'Convsent_{model_striped}_{args.editing_method}_sft_results_{args.add2path}.json'), 'w'), indent=4)
        else:
            json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'Convsent_{model_striped}_{args.editing_method}_results_{args.add2path}.json'), 'w'), indent=4)
    else:
        json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'Convsent_{model_striped}_{args.editing_method}_results_{args.add2path}.json'), 'w'), indent=4)
