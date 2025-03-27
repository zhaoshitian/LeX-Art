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
#### (1) torch
```bash
bash scripts/sample_text.sh
```
#### (2) diffuser
```python
import torch
from diffusers import Lumina2Pipeline

pipe = Lumina2Pipeline.from_pretrained("X-ART/LeX-Lumina", torch_dtype=torch.bfloat16)
pipe.to("cuda")
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "The image features a bold, dramatic design centered around the text elements \"THE,\" \"RA,\" and \"SA4GONEARAz,\" arranged to form the title of *The Boulet Brothers Dragula Season Three*. The background is a textured, dark slate-gray surface with faint grunge patterns, adding a gritty, industrial vibe. The word \"THE\" is positioned at the top in large, jagged, blood-red letters with a glossy finish and slight drop shadows, evoking a horror-inspired aesthetic. Below it, \"RA\" appears in the middle-left section, rendered in metallic silver with a fragmented, cracked texture, while \"SA4GONEARAz\" curves dynamically to the right, its letters styled in neon-green and black gradients with angular, cyberpunk-inspired edges. The number \"4\" in \"SA4GONEARAz\" replaces an \"A,\" blending seamlessly into the stylized typography. Thin, glowing purple outlines highlight the text, contrasting against the dark backdrop. Subtle rays of violet and crimson light streak diagonally across the composition, casting faint glows around the letters. The overall layout balances asymmetry and cohesion, with sharp angles and a mix of organic and mechanical design elements, creating a visually intense yet polished aesthetic that merges gothic horror with futuristic edge."
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=50,
    cfg_trunc_ratio=1,
    cfg_normalization=True,
    generator=torch.Generator("cpu").manual_seed(0),
    system_prompt="You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts.",

).images[0]
image.save("lex_lumina_demo.png")
```
