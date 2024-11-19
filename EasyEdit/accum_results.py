import json

# Load the data
with open('/workspace/DPO_research/EasyEdit/examples/output/<your_output_file>.json', 'r') as f:
    data = json.load(f)

# Initialize accumulators
rewrite_acc_sum = 0
locality_sum = 0
portability_sum = 0
rephrase_acc_sum = 0
count = 0

# Accumulate values
for item in data:
    post = item['post']
    rewrite_acc_sum += post['rewrite_acc'][0]
    locality_sum += post['locality']['neighborhood_acc'][0]
    portability_sum += post['portability']['one_hop_acc'][0]
    rephrase_acc_sum += post['rephrase_acc'][0]
    count += 1

# Calculate averages
rewrite_acc_avg = rewrite_acc_sum / count
locality_avg = locality_sum / count
portability_avg = portability_sum / count
rephrase_acc_avg = rephrase_acc_sum / count

# Print averages
print(f'Average rewrite_acc: {rewrite_acc_avg}')
print(f'Average locality: {locality_avg}')
print(f'Average portability: {portability_avg}')
print(f'Average rephrase_acc: {rephrase_acc_avg}')
