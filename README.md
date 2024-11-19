# Knowledge Editing in Language Models via Adapted Direct Preference Optimization

This repository contains the official code implementation for the paper **"Knowledge Editing in Language Models via Adapted Direct Preference Optimization"** published in **EMNLP (Findings) 2024**. The paper proposes a novel approach for knowledge editing in pre-trained language models, leveraging adapted direct preference optimization to modify model knowledge in a controlled manner.

Paper: [Knowledge Editing in Language Models via Adapted Direct Preference Optimization (arXiv)](https://arxiv.org/abs/2406.09920)

## Installation

### Prerequisites

Make sure you have the following dependencies installed:

- Python >= 3.9
- PyTorch 2.0.1
- CUDA 
- Code was tested on an NVIDIA A100 80GB GPU
- Additional dependencies specified in `requirements.txt`

### Step-by-step Installation

1. Clone this repository

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage

### Training KDPO on ZsRE dataset
Example command (assuming running from /path/to/EasyEdit/examples):
```bash
python EasyEdit/examples/run_zsre_llama2.py --editing_method=DPO --hparams_dir=../hparams/DPO/llama-7b.yaml --data_dir=../../data
```

## Evaluation 
Loads the results json file from training. 

Example evaluation command:
```bash
python EasyEdit/accum_results.py
```

## Acknowledgments

- **EasyEdit**: This code is heavily inspired by the [EasyEdit repository](https://github.com/zjunlp/EasyEdit) by the ZJUNLP team. We thank them for their excellent work and contribution to knowledge editing research.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---
