import requests
import torch
from transformers import AutoModelForCausalLM
import json
import os
from PIL import Image
from tqdm import tqdm

path = os.getenv("BENCH_RESULT_PATH")
all_data_list = json.load(open(path, "r", encoding="utf-8"))

model = AutoModelForCausalLM.from_pretrained("q-future/one-align", trust_remote_code=True, attn_implementation="eager", 
                                             torch_dtype=torch.float16, device_map="auto")


new_data_list = []
for item in tqdm(all_data_list):
    image_path = item['simple_image_path']
    enhanced_image_path = item['enhanced_image_path']
    image = Image.open(image_path).convert("RGB")
    enhanced_image = Image.open(enhanced_image_path).convert("RGB")
    qscore_image = model.score([image], task_="quality", input_="image").cpu().tolist()
    qscore_enhanced_image = model.score([enhanced_image], task_="quality", input_="image").cpu().tolist()
    ascore_image = model.score([image], task_="aesthetics", input_="image").cpu().tolist()
    ascore_enhanced_image = model.score([enhanced_image], task_="aesthetics", input_="image").cpu().tolist()
    # task_ : quality | aesthetics; # input_: image | video

    item['q_score_simple_image'] = qscore_image[0]
    item['q_score_enhanced_image'] = qscore_enhanced_image[0]
    item['a_score_simple_image'] = ascore_image[0]
    item['a_score_enhanced_image'] = ascore_enhanced_image[0]

    new_data_list.append(item)


save_path = os.getenv("SAVE_PATH")
with open(save_path, "w") as f:
    json.dump(new_data_list, f, indent=4)
