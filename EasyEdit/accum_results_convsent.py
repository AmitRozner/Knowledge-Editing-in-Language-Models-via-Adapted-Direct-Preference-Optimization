import json

# Load the data
#with open('/workspace/DPO_research/EasyEdit/examples/output/ROME_results.json', 'r') as f:

#with open('/workspace/DPO_research/EasyEdit/examples/output/KN_results.json', 'r') as f:
#with open('/workspace/DPO_research/EasyEdit/examples/output/ROME_results_full.json', 'r') as f:



def accum_states(data, file_name=""):
    global item
    # Initialize accumulators

    es_sum = 0
    fluency_sum = 0
    count = 0
    # Accumulate values
    for item in data:
        if "post" not in item:
            continue
        es_sum += item['es']
        fluency_sum += item['fluency']['ngram_entropy']
        count += 1
    # Calculate averages
    es_avg = es_sum / count
    fluency_avg = fluency_sum / count


    # Print averages
    print(f'Average es: {es_avg}')
    print(f'Average fluency: {fluency_avg}')


    metrics =  {
    "Average es": es_avg,
    "Average fluency": fluency_avg}

    json.dump(metrics, open(file_name, 'w'), indent=4)

if __name__ == "__main__":
    file = '/workspace/DPO_research/EasyEdit/examples/output/DPO_counterfact__on+offline_results.json'
    file = '/workspace/DPO_research/EasyEdit/examples/output/DPO_counterfact_Llama-2-7b-hf_results.json'
    output_file = '/workspace/DPO_research/EasyEdit/examples/output/DPO_counterfact_Llama-2-7b-hf_results_final.json'
    with open(file, 'r') as f:
        data = json.load(f)
    accum_states(data, file_name=output_file)
