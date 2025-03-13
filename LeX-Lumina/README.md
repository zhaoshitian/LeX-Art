## Installation
### 1. Create a conda environment and install PyTorch
```bash
conda create -n lumina2 -y
conda activate lumina2
conda install python=3.11 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install torch==2.1.0+cu121 torchvision==0.16.0 torchaudio==2.1.0 -f https://download.pytorch.org/whl/torch_stable.html
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Install flash-attn
```bash
pip install flash-attn --no-build-isolation
```

## Prepare data
You can place the links to your data files in `./configs/data.yaml`. Your image-text pair training data format should adhere to the following:
```json
{
    "image_path": "path/to/your/image",
    "prompt": "a description of the image"
}
```
## Train and Inference 
### 1. Finetuning
```bash
bash scripts/sft_lex_lumina_text.sh
```

### 2. Direct Inference
```bash
bash scripts/sample_text.sh
```


