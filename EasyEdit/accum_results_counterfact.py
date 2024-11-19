import json

# Load the data
#with open('/workspace/DPO_research/EasyEdit/examples/output/ROME_results.json', 'r') as f:

#with open('/workspace/DPO_research/EasyEdit/examples/output/KN_results.json', 'r') as f:
#with open('/workspace/DPO_research/EasyEdit/examples/output/ROME_results_full.json', 'r') as f:



def accum_states(data, file_name=""):
    global item
    # Initialize accumulators
    rewrite_acc_sum = 0
    locality_sum = 0
    locality_forget_sum = 0
    portability_sum = 0
    portability_sum_reason = 0
    portability_sum_logic = 0
    fluency_acc_sum = 0
    count = 0
    count_loc = 0
    count_loc_forget = 0
    count_port = 0
    count_port_reas = 0
    count_port_logic = 0
    count_flu = 0
    # Accumulate values
    for j,item in enumerate(data):
        if "post" not in item:
            continue
        post = item['post']
        rewrite_acc_sum += post['rewrite_acc'][0]
        if "EQ_DPO_wikibio_Llama-2-7b-hf_regdporegsample_stopcrit0.001_lr0.0001_mask_False_cyc5_numsteps6_nllterm_False_layer21_results_100" in file:
            print(j,":",post['rewrite_acc'][0])
        try:
            loc = post['locality']['Relation_Specificity_acc']
            locality_sum += sum(loc)/len(loc)
            count_loc += 1
        except:
            pass
        try:
            loc_forget = post['locality']['Forgetfulness_acc']
            locality_forget_sum += sum(loc_forget)/len(loc_forget)
            count_loc_forget += 1
        except:
            pass
        try:
            portability = post['portability']['Subject_Aliasing_acc']
            portability_sum += sum(portability)/len(portability)
            count_port += 1
        except:
            pass
        try:
            portability_reason = post['portability']['reasoning_acc']
            portability_sum_reason += sum(portability_reason)/len(portability_reason)
            count_port_reas += 1
        except:
            pass
        try:
            portability_logic = post['portability']['Logical_Generalization_acc']
            portability_sum_logic += sum(portability_logic)/len(portability_logic)
            count_port_logic += 1
        except:
            pass
        try:
            fluency_acc_sum += post['fluency']["ngram_entropy"]
            count_flu += 1
        except:
            pass
        count += 1
    # Calculate averages
    rewrite_acc_avg = rewrite_acc_sum / count
    if count_loc!=0:
        locality_avg = locality_sum / count_loc
    else:
        locality_avg = 0
    if count_loc_forget!=0:
        locality_forget_avg = locality_forget_sum / count_loc_forget
    else:
        locality_forget_avg = 0
    if count_port != 0:
        portability_avg = portability_sum / count_port
    else:
        portability_avg = 0
    if count_port_reas!=0:
        portability_reason_avg = portability_sum_reason / count_port_reas
    else:
        portability_reason_avg = 0
    if count_port_logic!=0:
        portability_logic_avg = portability_sum_logic / count_port_logic
    else:
        portability_logic_avg = 0
    if count_flu!=0:
        fluency_acc_avg = fluency_acc_sum / count_flu
    else:
        fluency_acc_avg = 0

    # Print averages
    print(f'Average rewrite_acc: {rewrite_acc_avg}')
    print(f'Average locality: {locality_avg}')
    print(f'Average locality Forgetfullness: {locality_forget_avg}')
    print(f'Average portability: {portability_avg}')
    print(f'Average portability reason: {portability_reason_avg}')
    print(f'Average portability logic: {portability_logic_avg}')
    print(f'Average fluency: {fluency_acc_avg}')
    metrics =  {
    "Average rewrite_acc": rewrite_acc_avg,
    "Average locality": locality_avg,
    "Average Forgetfullness locality": locality_forget_avg,
    "Average portability": portability_avg,
    "Average portability reason": portability_reason_avg,
    "Average portability logic": portability_logic_avg,
    "Average fluency": fluency_acc_avg}

    json.dump(metrics, open(file_name, 'w'), indent=4)

if __name__ == "__main__":
    file_bool = False
    if file_bool:
        file = "/workspace/DPO_research/EasyEdit/examples/output/SEQ_DPO_recent_Qwen1.5-0.5B_debug_stopcrit0.001_results_00.json"


        output_file = file.replace(".json","_final.json")#'/workspace/DPO_research/EasyEdit/examples/output/ROME_counterfact_Llama-2-7b-hf_results_final.json'
        with open(file, 'r') as f:
            data = json.load(f)
        accum_states(data, file_name=output_file)
    else:
        dir = "/workspace/DPO_research/EasyEdit/examples/output/lr1e-4_8steps_20cycles_afterfix/"
        #dir = "/workspace/DPO_research/EasyEdit/examples/output/lr1e-4_8steps_20cycles_afterfix_500"
        import  glob
        for file in glob.glob(dir+"/*"):
            if "final" in file:
                continue
            print("File:",file)

            output_file = file.replace(".json",
                                       "_final.json")  # '/workspace/DPO_research/EasyEdit/examples/output/ROME_counterfact_Llama-2-7b-hf_results_final.json'
            with open(file, 'r') as f:
                data = json.load(f)
            accum_states(data, file_name=output_file)

