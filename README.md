<div align="center">


# LeX-Art: Rethinking Visual Text Generation from Complex Glyph Modules to Scalable High-Quality Data Synthesis.

[[Paper]()] &emsp; [[Project Page](https://zhaoshitian.github.io/lexart/)] &emsp; 

[[🤗Model Card (LeX-Enhancer)](https://huggingface.co/X-ART/LeX-Enhancer-full)] &emsp; [[🤗Model Card (LeX-Lumina)](https://huggingface.co/X-ART/LeX-Lumina)] &emsp; [[🤗Data (LeX-Data)](https://huggingface.co/datasets/X-ART/LeX-Data-10K)]  &emsp; [[🤗Bench (LeX-Bench)](https://huggingface.co/datasets/X-ART/LeX-Bench)] <br>

</div>

This is the official repository for **LeX-Art: Rethinking Visual Text Generation from Complex Glyph Modules to Scalable High-Quality Data Synthesis**.

### 🌠  **Key Features:**

1. Proposed **LeX-Art**, a system bridging prompt expressiveness and text rendering fidelity.
2. Curated **LeX-10K**, a dataset of 10K high-resolution (1024×1024) aesthetically refined images.
3. Developed **LeX-Enhancer** for prompt enrichment and trained two text-to-image models, **LeX-FLUX** and **LeX-Lumina**.
4. Introduced **LeX-Bench** for evaluating fidelity, aesthetics, and alignment, along with the **Pairwise Normalized Edit Distance (PNED) metric** for text accuracy.





## 🎤 Introduction

Generating visually appealing and accurate text within images is challenging due to the difficulty of balancing text fidelity, aesthetic integration, and stylistic diversity. To address this, we introduce **LeX**, a framework that enhances text-to-image generation through **LeX-Enhancer**, a 14B-parameter prompt optimizer, and **LeX-10K**, a high-quality dataset. Using this, we train **LeX-Flux (12B)** and **LeX-Lumina (2B)**, achieving state-of-the-art performance. We also propose **LeX-Bench** and **PNED**, a novel metric for evaluating text correctness and aesthetics. Experiments show **LeX-Lumina** improves PNED by **22.16%**, while **LeX-Flux** enhances color accuracy, position accuracy, and font fidelity.

## 📬 News

- ✅ March 27, 2025. 💥 We release **LeX-Art**, including:
  - Checkpoints, Inference and Evaluate code.
  - Website.


## 🔥 Gallery

### Demos

![demos1](./assets/demos1.png "demos1")

![demos0](./assets/demos0.png "demos0")

### Samples from LeX-10K

![train_data](./assets/train_data.jpg "train_data")


## 📁 Data Synthesis

![overview](./assets/overview.jpg "overview")


## 📊 Experimental Results

![results](./assets/results.png "results")

## 🚀 Getting Started

### 🛠️ Installation

```bash
git clone https://github.com/zhaoshitian/LeX-Art.git
cd LeX-Art
conda create -n lex python=3.10

# if cuda version == 12.1
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


### 📥 Download Models

(Provide instructions for downloading models here)


### 🔍 Inference

We provide multiple tools for inference, tailored to different tasks:

- **[LeX-Enhancer](https://github.com/zhaoshitian/LeX-Art/blob/main/LeX-Enhancer/README.md)**: A tool designed to enhance prompts for improved text-to-image generation.
- **[LeX-Lumina](https://github.com/zhaoshitian/LeX-Art/blob/main/LeX-Lumina/README.md)**: A text-to-image (T2I) model further trained on Lumina-Image-2.0, capable of generating high-quality images with precise text rendering from prompts.
- **[LeX-FLUX](https://github.com/zhaoshitian/LeX-Art/blob/main/LeX-FLUX/README.md)**: A text-to-image (T2I) model further trained on FLUX.1, capable of generating high-quality images with precise text rendering from prompts.

Click on the links above for detailed instructions on how to use each tool.


### 📊 Evaluation

For detailed instructions on model evaluation, please refer to the [Evaluation README](https://github.com/zhaoshitian/LeX-Art/blob/main/evaluation/README.md).


## 📌 Open-source Plan

- [x] Release the inference code.
- [x] Release the evaluation code.
- [x] Release the data and checkpoints for LeX Series.
- [ ] Release the training code.

## 📚 BibTeX

If you find LeX-Art useful for your research and applications, please cite using this BibTeX:

```BibTeX
@article{zhao2025lexart,
  title={LeX-Art: Rethinking Text Generation for Visual Content via Scalable High-Quality Data Synthesis},
  author={Zhao, Shitian and Wu, Qilong and Li, Xinyue and Zhang, Bo and Li, Ming and Qin, Qi and Liu, Dongyang and Zhang, Kaipeng and Gao, Peng and Fu, Bin and Li, Zhen},
  journal={arXiv preprint},
  year={2025}
}
```