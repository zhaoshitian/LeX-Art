import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("X-ART/LeX-FLUX", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "The image portrays a vibrant scene of a woman passionately singing into a handheld silver microphone, her head slightly tilted back and her auburn hair flowing as if caught mid-motion. She stands confidently on a circular stage, bathed in warm amber and cool blue spotlights that create a dynamic contrast and soft shadows. Her outfit, a sleek black leather jacket paired with matching fitted pants and high-heeled boots, complements her commanding presence. Behind her, the text "NEW ARTISTS" appears prominently in bold gold at the top, while "ARTI 2021" is rendered in glowing neon-blue at the bottom, with some text artifacts present as "ARTI" appears incomplete. The background features a gradient of deep indigo to muted teal with subtle glowing particles scattered throughout, adding texture and depth. The overall composition captures a lively, contemporary aesthetic, blending realism with stylized elements, and highlights the energy and focus of the performance despite minor visual inconsistencies in the text rendering."
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("demo.png")
