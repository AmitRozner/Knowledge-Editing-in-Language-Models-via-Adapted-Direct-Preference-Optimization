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
    SERACHparams,

    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--add2path', required=False,default="", type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--datatype', default="zsre", type=str)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--seq', action="store_true")

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
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'DPO':
        editing_hparams = DPOHyperParams
    else:
        raise NotImplementedError
    #if args.editing_method == 'DPO':
    #    test_data = json.load(open(os.path.join(args.data_dir, 'zsre_mend_eval_portability_gpt4_dpo.json'), 'r', encoding='utf-8'))
    #else:

    test_data = json.load(open(os.path.join(args.data_dir, 'zsre_mend_eval_portability_gpt4.json'), 'r', encoding='utf-8'))
    # for in_ in test_data:
    #     list_ =in_["dpo_incorrect_answers"]
    #     if "[" in list_ and "]" not in list_:
    #         print("stop here")


    if args.ds_size is not None:
        test_data = random.sample(test_data, args.ds_size)
    test_data = test_data#[4:6]#[800:900]#[867:870]

    prompts = [test_data_['src'] for test_data_ in test_data]
    rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in test_data]
    target_new = [edit_data_['alt'] for edit_data_ in test_data]
    locality_prompts = [edit_data_['loc'] for edit_data_ in test_data]
    locality_ans = [edit_data_['loc_ans'] for edit_data_ in test_data]
    portability_prompts = [edit_data_['portability']['New Question'] for edit_data_ in test_data]
    portability_ans = [edit_data_['portability']['New Answer'] for edit_data_ in test_data]

    dpo_data=None
    locality_inputs = {
        'neighborhood':{
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }
    portability_inputs = {
        'one_hop':{
            'prompt': portability_prompts,
            'ground_truth': portability_ans
        },
    }
    subject = [edit_data_['subject'] for edit_data_ in test_data]
    hparams = editing_hparams.from_hparams(args.hparams_dir)

    if args.editing_method == 'IKE':
        train_data_path = os.path.join(args.data_dir, 'zsre_mend_train_10000.json')
        train_ds = ZsreDataset(train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None

    editor = BaseEditor.from_hparams(hparams)
    if args.seq:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            subject=subject,
            train_ds=train_ds,
            dpo_data=dpo_data,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            args=args,
            keep_original_weight=False
        )
    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            subject=subject,
            train_ds=train_ds,
            dpo_data=dpo_data,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            args=args,
            keep_original_weight=True
        )

    model_striped = args.hparams_dir.split("/")[-1].replace(".yaml","")
    if args.seq:
        json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'SEQ_{args.editing_method}_ZSRE_{hparams.model_name.split("/")[-1]}_{args.add2path}_results.json'), 'w'), indent=4)
    else:
        json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_ZSRE_{hparams.model_name.split("/")[-1]}_{args.add2path}_results.json'), 'w'), indent=4)
