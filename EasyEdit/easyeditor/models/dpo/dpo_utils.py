import torch

def find_and_replace_identical_blocks(tensor1, tensor2, min_block_size=4, replace_value=-100):
    max_len = min(tensor1.size(-1), tensor2.size(-1))
    identical_blocks = []

    # Iterate over all possible sub-array sizes, from largest to smallest
    for size in range(max_len, min_block_size - 1, -1):
        # Slide over the tensor with the given size
        for start in range(max_len - size + 1):
            subblock1 = tensor1[:,start:start + size]
            # Check if this subblock exists in tensor2
            for start2 in range(max_len - size + 1):
                subblock2 = tensor2[:,start2:start2 + size]
                if torch.equal(subblock1, subblock2):
                    identical_blocks.append((start, start2, size))
                    break  # exit inner loop early upon finding a match
            else:
                continue  # only executed if the inner loop did NOT break
            break  # exit outer loop early upon finding a match

    # Replace identified blocks in both tensors
    for start1, start2, size in identical_blocks:
        tensor1[:,start1:start1 + size] = replace_value
        tensor2[:,start2:start2 + size] = replace_value

    return tensor1, tensor2

