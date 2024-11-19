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
    portability_sum = 0
    portability_sum_reason = 0
    fluency_acc_sum = 0
    count = 0
    count_loc = 0
    count_port = 0
    count_port_reas = 0
    count_flu = 0
    # Accumulate values
    for item in data:
        if "post" not in item:
            continue
        post = item['post']
        rewrite_acc_sum += post['rewrite_acc'][0]
        try:
            locality_sum += post['locality']['Relation_Specificity_acc'][0]
            count_loc += 1
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
    if count_port != 0:
        portability_avg = portability_sum / count_port
    else:
        portability_avg = 0

    if count_port_reas!=0:
        portability_reason_avg = portability_sum_reason / count_port_reas
    else:
        portability_reason_avg = 0
    if count_flu!=0:
        fluency_acc_avg = fluency_acc_sum / count_flu
    else:
        fluency_acc_avg = 0

    # Print averages
    print(f'Average rewrite_acc: {rewrite_acc_avg}')
    print(f'Average locality: {locality_avg}')
    print(f'Average fluency: {fluency_acc_avg}')
    metrics =  {
    "Average rewrite_acc": rewrite_acc_avg,
    "Average locality": locality_avg,
    "Average fluency": fluency_acc_avg}

    json.dump(metrics, open(file_name, 'w'), indent=4)

if __name__ == "__main__":
    file = '/workspace/DPO_research/EasyEdit/examples/output/DPO_wikibio_Llama-2-7b-hf_decrease_lr_2loops_results.json'
    #file = '/workspace/DPO_research/EasyEdit/examples/output/ROME_counterfact_Llama-2-7b-hf_results.json'
    output_file = file.replace(".json","_final.json")#'/workspace/DPO_research/EasyEdit/examples/output/ROME_counterfact_Llama-2-7b-hf_results_final.json'
    with open(file, 'r') as f:
        data = json.load(f)
    accum_states(data, file_name=output_file)
