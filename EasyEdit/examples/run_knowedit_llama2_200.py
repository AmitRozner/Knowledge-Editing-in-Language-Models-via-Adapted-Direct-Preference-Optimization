import os
import os.path
import sys
import json
import random
sys.path.append('..')
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
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import KnowEditDataset

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--datatype', default=None,type=str)
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--pre_file', default='./seq_pre.json', type=str)
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
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'DPO':
        editing_hparams = DPOHyperParams
    else:
        raise NotImplementedError
    

    datas = KnowEditDataset(args.data_dir,size=args.ds_size,args=args)
    datas = datas[:200]
    if args.datatype == 'counterfact' or args.datatype == 'recent' or args.datatype == 'zsre':
        prompts=[data['prompt'] for data in datas]
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        
        portability_r =[data['portability_r'] for data in datas]
        portability_s =[data['portability_s'] for data in datas]
        portability_l =[data['portability_l'] for data in datas]

        portability_reasoning_prompts=[]
        portability_reasoning_ans=[]
        portability_Logical_Generalization_prompts=[]
        portability_Logical_Generalization_ans=[]
        portability_Subject_Aliasing_prompts=[]
        portability_Subject_Aliasing_ans=[]
        
        portability_data = [portability_r,portability_s,portability_l]
        portability_prompts = [portability_reasoning_prompts,portability_Subject_Aliasing_prompts,portability_Logical_Generalization_prompts]
        portability_answers = [portability_reasoning_ans,portability_Subject_Aliasing_ans,portability_Logical_Generalization_ans]
        for data, portable_prompts, portable_answers in zip(portability_data,portability_prompts,portability_answers):
            for item in data:
                if item is None:
                    portable_prompts.append(None)
                    portable_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        while isinstance(an,list):
                            an = an[0]
                        if an.strip() =="":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    portable_prompts.append(temp_prompts)
                    portable_answers.append(temp_answers)
        assert len(prompts) == len(portability_reasoning_prompts) == len(portability_Logical_Generalization_prompts) == len(portability_Subject_Aliasing_prompts)
        
        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]
        locality_Relation_Specificity_prompts=[]
        locality_Relation_Specificity_ans=[]
        locality_Forgetfulness_prompts=[]        
        locality_Forgetfulness_ans=[]
        
        locality_data = [locality_rs, locality_f]
        locality_prompts = [locality_Relation_Specificity_prompts,locality_Forgetfulness_prompts]
        locality_answers = [locality_Relation_Specificity_ans,locality_Forgetfulness_ans]
        for data, local_prompts, local_answers in zip(locality_data,locality_prompts,locality_answers):
            for item in data:
                if item is None:
                    local_prompts.append(None)
                    local_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        while isinstance(an,list):
                            an = an[0]
                        if an.strip() =="":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    local_prompts.append(temp_prompts)
                    local_answers.append(temp_answers)
        assert len(prompts) == len(locality_Relation_Specificity_prompts) == len(locality_Forgetfulness_prompts)
        locality_inputs = {}
        portability_inputs = {}
        
        locality_inputs = {
            'Relation_Specificity':{
                'prompt': locality_Relation_Specificity_prompts,
                'ground_truth': locality_Relation_Specificity_ans
            },
            'Forgetfulness':{
                'prompt':locality_Forgetfulness_prompts,
                'ground_truth':locality_Forgetfulness_ans
            }
        }
        portability_inputs = {
            'Subject_Aliasing':{
                'prompt': portability_Subject_Aliasing_prompts,
                'ground_truth': portability_Subject_Aliasing_ans
            },
            'reasoning':{
                'prompt': portability_reasoning_prompts,
                'ground_truth': portability_reasoning_ans           
            },
            'Logical_Generalization':{
                'prompt': portability_Logical_Generalization_prompts,
                'ground_truth': portability_Logical_Generalization_ans           
            }
        }
    elif args.datatype == 'wikibio':
        prompts=[data['prompt'] for data in datas]
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        
        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]
        locality_Relation_Specificity_prompts=[]
        locality_Relation_Specificity_ans=[]
        
        locality_data = [locality_rs]
        locality_prompts = [locality_Relation_Specificity_prompts]
        locality_answers = [locality_Relation_Specificity_ans]
        for data, local_prompts, local_answers in zip(locality_data,locality_prompts,locality_answers):
            for item in data:
                if item is None:
                    local_prompts.append(None)
                    local_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        while isinstance(an,list):
                            an = an[0]
                        if an.strip() =="":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    local_prompts.append(temp_prompts)
                    local_answers.append(temp_answers)
        assert len(prompts) == len(locality_Relation_Specificity_prompts)
        portability_inputs = None
        locality_inputs = {}
        locality_inputs = {
            'Relation_Specificity':{
                'prompt': locality_Relation_Specificity_prompts,
                'ground_truth': locality_Relation_Specificity_ans
            }
        }


    dpo_data = None
    # dpo_data_ = dpo_data
    # request={}
    # for j,dp_data in enumerate(dpo_data_):
    #     print("j:",j)
    #     request["dpo_data"] = dp_data.replace("\n","")
    #     if "[" in request["dpo_data"] and "]" not in request["dpo_data"]:
    #         if '"' in request["dpo_data"]:
    #             request["dpo_data"] = request["dpo_data"] + '"]'
    #         elif "'" in request["dpo_data"]:
    #             request["dpo_data"] = request["dpo_data"] + "']"
    #         else:
    #             request["dpo_data"] = request["dpo_data"] + "]"
    #     for in_ in dpo_data_:
    #         request["dpo_data"] = request["dpo_data"].replace("'", "")
    #         try:
    #             dpo_data = eval(request["dpo_data"])
    #         except:
    #             try:
    #                 dpo_data = request["dpo_data"].replace("[", "['").replace("]", "']").replace(",", "','")
    #                 dpo_data = eval(dpo_data)
    #             except:
    #                 dpo_data = eval(request["dpo_data"].replace('’', '"').replace('‘', '"'))

    hparams = editing_hparams.from_hparams(args.hparams_dir)
    if args.ds_size is not None:
        args.pre_file = f"./{hparams.model_name.split('/')[-1]}_{args.datatype}_{args.ds_size}pre_edit_200.json"
    else:
        args.pre_file = f"./{hparams.model_name.split('/')[-1]}_{args.datatype}_pre_edit_200.json"
    print(args.pre_file)
    if  False and args.pre_file is not None and os.path.exists(args.pre_file):
        pre_edit = json.load(open(args.pre_file,'r'))
        if len(pre_edit) != len(prompts):
           pre_edit = None
    else:
        pre_edit = None
    if args.editing_method == 'IKE':
        train_ds = KnowEditDataset(args.train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None
    editor = BaseEditor.from_hparams(hparams)
    if args.seq:
        metrics, edited_model, _ = editor.seq_edit(
            prompts=prompts,
            target_new=target_new,
            subject=subjects,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            train_ds=train_ds,
            dpo_data=dpo_data,
            keep_original_weight=False,
            pre_file=args.pre_file,
            pre_edit=pre_edit,
            args=args,
            test_generation=True,
        )
    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            subject=subjects,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            train_ds=train_ds,
            dpo_data=dpo_data,
            keep_original_weight=True,
            pre_file=args.pre_file,
            pre_edit = pre_edit,
            args=args,
            test_generation=True,
        )
    model_striped = args.hparams_dir.split("/")[-1].replace(".yaml","")
    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    if args.seq:
        try:
            json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'SEQ_{args.editing_method}_{args.datatype}_{hparams.model_name.split("/")[-1]}_{args.add2path}_stopcrit{hparams.stop_crit}_results_200.json'), 'w'), indent=4)
        except:
            json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'SEQ_{args.editing_method}_{args.datatype}_{hparams.model_name.split("/")[-1]}_{args.add2path}_results_200.json'), 'w'), indent=4)
    else:
        if args.editing_method=="DPO":
            if hparams.use_identical_blocks:
                json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_{args.datatype}_{hparams.model_name.split("/")[-1]}_{args.add2path}_identicalblocks_results.json'), 'w'), indent=4)
            else:
                json.dump(metrics, open(os.path.join(args.metrics_save_dir,
                                                     f'{args.editing_method}_{args.datatype}_{hparams.model_name.split("/")[-1]}_{args.add2path}_results.json'),
                                        'w'), indent=4)

        else:
            json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_{args.datatype}_{hparams.model_name.split("/")[-1]}_{args.add2path}_results.json'), 'w'), indent=4)
