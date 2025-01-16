# DemoNSF: A Multi-task Demonstration-based Generative Framework for Noisy Slot Filling Task


## 🎥 Overview
This repository contains the open-sourced official implementation of the paper:

[DemoNSF: A Multi-task Demonstration-based Generative Framework for Noisy Slot Filling Task](https://arxiv.org/pdf/2310.10169v1.pdf) (Findings of EMNLP 2023 (Short Paper)).


If you find this repo helpful, please cite the following paper:

```bibtex
@inproceedings{dong2023demonsf,
  author       = {Guanting Dong and
                  Tingfeng Hui and
                  Zhuoma Gongque and
                  Jinxu Zhao and
                  Daichi Guo and
                  Gang Zhao and
                  Keqing He and
                  Weiran Xu},
  editor       = {Houda Bouamor and
                  Juan Pino and
                  Kalika Bali},
  title        = {DemoNSF: {A} Multi-task Demonstration-based Generative Framework for
                  Noisy Slot Filling Task},
  booktitle    = {Findings of the Association for Computational Linguistics: {EMNLP}
                  2023, Singapore, December 6-10, 2023},
  pages        = {10506--10518},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://doi.org/10.18653/v1/2023.findings-emnlp.705},
  doi          = {10.18653/V1/2023.FINDINGS-EMNLP.705},
  timestamp    = {Sun, 06 Oct 2024 21:00:49 +0200},
  biburl       = {https://dblp.org/rec/conf/emnlp/DongHGZGZHX23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}



```

## Introduction
We propose a Multi-task **Demo**nstration based Generative Framework for **N**oisy **S**lot **F**illing, named DemoNSF. Specifically, we introduce three noisy auxiliary tasks, namely **Noisy Recovery (NR)**, **Random Mask (RM)**, and **Hybrid Discrimination (HD)**, to implicitly capture semantic structural information of input perturbations at different granularities. In the downstream main task, we design a noisy demonstration construction strategy for the generative framework, which explicitly incorporates task-specific information and perturbed distribution during training and inference. Experiments on two benchmarks demonstrate that DemoNSF outperforms all baseline methods and achieves strong generalization. Further analysis provides empirical guidance for the practical application of generative frameworks. 

## 🍯 Overall Framework
<img width="700" alt="image" src="https://github.com/dongguanting/Demo-NSF/assets/60767110/0889d577-7b22-46f4-a074-44cb78da4ad8">

## 🎯 Quick Start

### Dependencies
```
conda create -n your_env_name python=3.9
conda activate your_env_name
pip3 install torch==2.0.1
pip3 install transformers==4.33.1
pip3 install sentence-transformers
pip3 install nlpaug
pip3 install datasets
```
You can copy and paste the above command in your terminal to create the environment.

### Datasets
The two benchmarks used in our paper are both under the data path, which are multiWOZ and multi-noise, respectively.

The process.py shows some data processing operations for the multi-noise dataset, including data augmentation, building demonstrations, converting to NER data, converting to mask data, converting to multi-classification data, and so on.

### Pre-training
```
bash scripts/pretrain.sh
```
Use the above command to start the pre-training stage.
```
--noise_path ${char_file} \
--clean_path ${train_file} \
--mask_output_path ${mask_output} \
--mask_input_path ${mask_input} \
--classify_output_path ${classify_output} \
--classify_input_path ${classify_input} \
```
The above parameters represent the datasets of the three pre-training tasks proposed in our pre-training stage.

### Training
```
bash scripts/train.sh
```
Use the above command to start the training stage.
```
--add_demonstration \
--demons_train_path ${mix_input} \
--demons_out_path ${mix_output} \
--demons_valid_path ${valid_demons_input} \
--demons_val_out_path ${valid_demons_output} \
```
The 'add_demonstration' represents whether to add demonstrations in the training stage. If this parameter is provided, the following four data paths related to the demonstration need to be provided.

### Testing
```
bash scripts/test.sh
```
Use the above command to start the testing stage.
```
--test_file_path ${test_path} \
```
The 'test_path' represents the root directory containing the test datasets.

