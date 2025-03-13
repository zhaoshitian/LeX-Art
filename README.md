# LeX-Art

This is the official repository for **LeX-Art: Rethinking Visual Text Generation from Complex Glyph Modules to Scalable High-Quality Data Synthesis**.

## Introduction
Generating visually appealing and accurate text within images is challenging due to the difficulty of balancing text fidelity, aesthetic integration, and stylistic diversity. To address this, we introduce **LeX**, a framework that enhances text-to-image generation through **LeX-Enhancer**, a 14B-parameter prompt optimizer, and **LeX-10K**, a high-quality dataset. Using this, we train **LeX-Flux (12B)** and **LeX-Lumina (2B)**, achieving state-of-the-art performance. We also propose **LeX-Bench** and **PNED**, a novel metric for evaluating text correctness and aesthetics. Experiments show **LeX-Lumina** improves PNED by **22.16%**, while **LeX-Flux** enhances color accuracy, position accuracy, and font fidelity.

## Installation
```bash
git clone https://github.com/zhaoshitian/LeX-Art.git
cd LeX-Art
conda create -n lex python=3.10
# if cuda version == 12.1
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Demos
![demos](./assets/demos.png "demos")

## Data Synthesis
![overview](./assets/overview.png "overview")

## Samples from LeX-10K
![train_data](./assets/train_data.jpg "train_data")

## Results Comparison
![results](./assets/results.png "results")

## Open-source Plan
- [X] Release the inference code.
- [X] Release the evaluation code.
- [X] Release the data and checkpoints for LeX Series.
- [ ] Release the training code.
