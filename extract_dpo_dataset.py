import os.path
import sys
sys.path.append('..')
import json
import random
from EasyEdit.easyeditor import (
    FTHyperParams,
    IKEHyperParams,
    KNHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams
    )
from EasyEdit.easyeditor import BaseEditor
from EasyEdit.easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from EasyEdit.easyeditor import ZsreDataset

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


#Things to ask :1. around each of the items in the list must be '', 2. if you write something with ' use \\'
#3. always note that you close ]
#
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=False, type=str)
    args = parser.parse_args()

    # Load model directly

    # pip install transformers bitsandbytes accelerate

    model_id = "CohereForAI/c4ai-command-r-v01-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_id)#, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id)#, trust_remote_code=True)


    #test_data = json.load(open(os.path.join(args.data_dir, 'zsre_mend_eval_portability_gpt4.json'), 'r', encoding='utf-8'))
    file = "/workspace/DPO_research/benchmark/wiki_counterfact/benchmark_wiki_counterfact_train_cf.json"
    out_file = "/workspace/DPO_research/benchmark/wiki_counterfact/benchmark_wiki_counterfact_train_cf_dpo.json"
    test_data = json.load(open( file, 'r', encoding='utf-8'))

    test_data = test_data
    #old = [test_data_['answer'] for test_data_ in test_data]
    prompts = [test_data_['prompt'] for test_data_ in test_data]
    target_new = [edit_data_['target_new'] for edit_data_ in test_data]

    outputs = []
    for j,(prompt,target) in enumerate(zip(prompts,    target_new)):
        print("##################################################################################")
        prompt = f"""I'm creating a multiple-choice quiz, and I need your help to generate some options. 
        You will be given a question followed by the correct answer. Your task is to provide ten plausible, yet incorrect, options. Ensure the options are relevant to the question and vary in nature. 
    
        Here's the first one:
    
        Question: {prompt}?
        Correct Answer: {target}
    
        Provide me with ten other options besides the correct one, and insert them to python list, do not use :’ or ` or ' or '.
        Further, dont use: '\n', dont use 'x for x in', dont explicitly write 'eval()' .
        What you extract goes right into eval() function in python, so think that it should pass successfully.
        Dont answer anything but the result themself, do not add numbers before each result."""



        # Format message with the command-r chat template
        messages = [{"role": "user", "content": prompt}]

        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,return_tensors="pt")
        ## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

        gen_tokens = model.generate(
            input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
        )

        gen_text = tokenizer.decode(gen_tokens[0])
        parsed = gen_text.split("CHATBOT_TOKEN|>")[1].split("<|END_OF_TURN_TOKEN|>")[0]
        print("First:",parsed)
        try:
            if isinstance(eval(parsed),list):
                segment = test_data[j]
                segment["dpo_incorrect_answers"] = parsed
                outputs.append(segment)
                continue
        except:
            pass
        prompt = f"""I'm creating a multiple-choice quiz, and I need your help to generate some options. 
        You will be given a question followed by the correct answer. Your task is to provide ten plausible, yet incorrect, options. Ensure the options are relevant to the question and vary in nature. 

        Here's the first one:

        Question: {prompt}?
        Correct Answer: {target}

        Provide me with ten other options besides the correct one, and insert them to python list, do not use :’ or ` or ' or '.
        Further, dont use: '\n', dont use 'x for x in', dont explicitly write 'eval()' .
        What you extract goes right into eval() function in python, so think that it should pass successfully.
        Dont answer anything but the result themself, do not add numbers before each result."""

        messages_make_sure = [{"role": "user", "content": prompt}]
        input_ids_makesure = tokenizer.apply_chat_template(messages_make_sure, tokenize=True, add_generation_prompt=True,return_tensors="pt")
        gen_tokens = model.generate(
            input_ids_makesure,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
        )
        gen_text = tokenizer.decode(gen_tokens[0])
        parsed = gen_text.split("CHATBOT_TOKEN|>")[1].split("<|END_OF_TURN_TOKEN|>")[0]
        print("Last:", parsed)
        segment = test_data[j]
        segment["dpo_incorrect_answers"] = parsed
        outputs.append(segment)
    with open(out_file, "w") as final:
        json.dump(outputs, final, indent=4)

        # prompt_full = f"""I'm creating a multiple-choice quiz, and I need your help to generate some options.
        # You will be given a question followed by the correct answer. Your task is to provide ten plausible, yet incorrect, options. Ensure the options are relevant to the question and vary in nature.
        #
        # Here's the first one:
        #
        # Question: {prompt}?
        # Correct Answer: {target}
        #
        # Provide me with ten other options besides the correct one."""

    #json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_results.json'), 'w'), indent=4)
