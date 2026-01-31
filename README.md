# FROST

## Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Data](#data)
- [Usage](#usage)
    - [Preprocessing](#1-preprocessing)
    - [Training](#2-training)
    - [Evaluation](#3-evaluation)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [Contact](#contact)

## Introduction
We propose FROST, an attention-aware method for efficient reasoning. Unlike traditional approaches, FROST leverages attention weights to prune uncritical reasoning paths, yielding shorter and more reliable reasoning trajectories. Methodologically, we introduce the concept of reasoning outliers and design an attention-based mechanism to remove them. Theoretically, FROST preserves and enhances the model’s reasoning capacity while eliminating outliers at the sentence level. Empirically, we validate FROST on four benchmarks using two strong reasoning models (Phi-4-Reasoning and GPT-oss-20B), outperforming state-of-the-art methods such as TALE and ThinkLess. Notably, FROST achieves an average 58.72% reduction in token usage and a 10.64% improvement in accuracy over the base model. Furthermore, in evaluations of attention outlier metrics, FROST reduces the maximum infinity norm 
 by 15.97% and the average kurtosis by 91.09% compared to the base model.

## Prerequisites
- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Other dependencies listed in `requirements.txt`

```bash
# create and activate virtual python environment
conda create -n frost python=3.10
conda activate frost

# install required packages
pip install -r requirements.txt
```

## Data
Please ensure your dataset is prepared and placed in the appropriate directory.
(Update this section with specific dataset instructions if available)

```
dataset/
```

## Usage

### 1. Preprocessing
(If there are specific preprocessing scripts, list them here. Based on the file list, you might need to run specific setup or data preparation steps.)

### 2. Training
You can train the model using the provided scripts. For experimental runs, use `train_sft.sh` which launches the training process.

```bash
# Run SFT training
bash train_sft.sh
```

Or run the python script directly:
```bash
python train_sft.py
```

### 3. Evaluation
Evaluate the trained model using pipeline listed in [EM_PT](https://github.com/shivamag125/EM_PT).

## TODO
- [ ] Add an attention-based outlier removal GRPO method.




## Acknowledgement
We appreciate the open-source community for their valuable code and efforts.
- [OutEffHop](https://github.com/MAGICS-LAB/OutEffHop)
- [ThinkLess](https://github.com/VainF/Thinkless)
- [TALE](https://github.com/GeniusHTX/TALE)
- [GERM](https://github.com/MAGICS-LAB/GERM)
- [EM_PT](https://github.com/shivamag125/EM_PT)
- [EasyR1](https://github.com/hiyouga/EasyR1)

## Citation
If you use FROST in your work, please kindly cite it:

```bibtex
@article{luo2026frost,
  title={FROST: Filtering Reasoning Outliers with Attention for Efficient Reasoning},
  author={Luo, Haozheng and Jiang, Zhuolin and Hasan, Md Zahid and Chen, Yan and Sarkar, Soumalya},
  journal={arXiv preprint arXiv:2601.19001},
  year={2026}
}
```

## Contact
If you have any questions or want to use the code, feel free to contact the authors.
