# DemoNSF: A Multi-task Demonstration-based Generative Framework for Noisy Slot Filling Task


## 🎥 Overview
This repository contains the open-sourced official implementation of the paper:

[DemoNSF: A Multi-task Demonstration-based Generative Framework for Noisy Slot Filling Task](https://arxiv.org/pdf/2310.10169v1.pdf) (Findings of EMNLP 2023 (Short Paper)).


If you find this repo helpful, please cite the following paper:

```bibtex
@misc{dong2023demonsf,
      title={DemoNSF: A Multi-task Demonstration-based Generative Framework for Noisy Slot Filling Task}, 
      author={Guanting Dong and Tingfeng Hui and Zhuoma GongQue and Jinxu Zhao and Daichi Guo and Gang Zhao and Keqing He and Weiran Xu},
      year={2023},
      eprint={2310.10169},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Introduction
We propose a Multi-task **Demo**nstration based Generative Framework for **N**oisy **S**lot **F**illing, named DemoNSF. Specifically, we introduce three noisy auxiliary tasks, namely **Noisy Recovery (NR)**, **Random Mask (RM)**, and **Hybrid Discrimination (HD)**, to implicitly capture semantic structural information of input perturbations at different granularities. In the downstream main task, we design a noisy demonstration construction strategy for the generative framework, which explicitly incorporates task-specific information and perturbed distribution during training and inference. Experiments on two benchmarks demonstrate that DemoNSF outperforms all baseline methods and achieves strong generalization. Further analysis provides empirical guidance for the practical application of generative frameworks. 

## 🍯 Overall Framework
<img width="700" alt="image" src="https://github.com/dongguanting/Demo-NSF/assets/60767110/0889d577-7b22-46f4-a074-44cb78da4ad8">

## 🎯 Quick Start
