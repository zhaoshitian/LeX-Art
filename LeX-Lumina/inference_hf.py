import torch
from diffusers import Lumina2Transformer2DModel, Lumina2Text2ImgPipeline

ckpt_path = "/mnt/hwfile/alpha_vl/qilongwu/checkpoints/Lumina-Image-2.0/consolidated.00-of-01.pth"
folder = "/mnt/hwfile/alpha_vl/qilongwu/checkpoints/Lumina-Image-2.0" # download change to "Alpha-VLLM/Lumina-Image-2.0"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = Lumina2Transformer2DModel.from_single_file(
    ckpt_path, torch_dtype=torch.bfloat16
)
pipe = Lumina2Text2ImgPipeline.from_pretrained(
    folder, transformer=transformer, torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

import time
# record time per prompt
start = time.time()
prompts = [
    "The Boulet Brothers Dragula Season Three, with text on it: \"THE\", \"RA\", \"SA4GONEARAz\"."
    # "A cat holding a sign that says hello.",
    # "A cat holding a sign that says goodbye.",
    # "A dragon flying over a volcano, breathing fire into the sky.",
    # "A giant robot standing on a city rooftop, overlooking the urban landscape.",
    # "A quiet village in the snow, with smoke rising from chimneys and a peaceful winter scene.",
    # "Cyberpunk cityscape at night, rainy street, neon signs with Chinese characters \"欢迎光临\", vibrant colors, reflections on wet pavement, futuristic atmosphere, detailed.",
    # "Watercolor painting of a traditional Chinese teahouse entrance, wooden sign hanging above the door with Chinese characters \"茶馆\" (teahouse), lanterns, peaceful garden setting, soft lighting.",
    # "The image features a bold, dramatic design centered around the text elements \"THE,\" \"RA,\" and \"SA4GONEARAz,\" arranged to form the title of *The Boulet Brothers Dragula Season Three*. The background is a textured, dark slate-gray surface with faint grunge patterns, adding a gritty, industrial vibe. The word \"THE\" is positioned at the top in large, jagged, blood-red letters with a glossy finish and slight drop shadows, evoking a horror-inspired aesthetic. Below it, \"RA\" appears in the middle-left section, rendered in metallic silver with a fragmented, cracked texture, while \"SA4GONEARAz\" curves dynamically to the right, its letters styled in neon-green and black gradients with angular, cyberpunk-inspired edges. The number \"4\" in \"SA4GONEARAz\" replaces an \"A,\" blending seamlessly into the stylized typography. Thin, glowing purple outlines highlight the text, contrasting against the dark backdrop. Subtle rays of violet and crimson light streak diagonally across the composition, casting faint glows around the letters. The overall layout balances asymmetry and cohesion, with sharp angles and a mix of organic and mechanical design elements, creating a visually intense yet polished aesthetic that merges gothic horror with futuristic edge.",
]
images = pipe(
    prompt=prompts,
    num_inference_steps=30,
    guidance_scale=4.0,
    negative_prompt=None,
    num_images_per_prompt=1,
    return_dict=True,
    system_prompt="You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts.",
    generator=torch.Generator("cpu").manual_seed(0),
    cfg_trunc_ratio=1,
    cfg_normalization=True,
    max_sequence_length=256,
).images
end = time.time()
print(f"Time taken: {end - start}")
print(f"Time per prompt: {(end - start) / len(prompts)}")

for i, img in enumerate(images):
    img.save(f"image_test{i}.png")
