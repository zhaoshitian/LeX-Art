import requests
import json
import base64
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from difflib import SequenceMatcher
import time
from requests.exceptions import ProxyError, RequestException
from tqdm import tqdm
import openai
from openai import OpenAI


thresh = os.getenv("THRESH")
img_source = os.getenv("IMG_SOURCE")
model_name = os.getenv("MODEL_NAME")
json_path = os.getenv("BENCH_RESULT_PATH")

Baseurl = "https://api.boyuerichdata.opensphereai.com/v1"
Skey = "sk-xxx"
client = OpenAI(api_key=Skey, base_url=Baseurl)
# Create output folder
output_folder = f"./benchmarks/results_vqa/cropped_images_{model_name}_{img_source}_{str(thresh)}"  # Replace with your output folder path
os.makedirs(output_folder, exist_ok=True)
output_json = f"./benchmarks/results_vqa/final_easy_{model_name}_{img_source}_{str(thresh)}.json"



# Define a function to calculate string similarity
def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Load JSON data
with open(json_path, "r") as f:  # Replace with your JSON file path
    data = json.load(f)

# Font style list
font_styles = [
    ["cursive style", "block style"],
    ["3D style", "flat style"],
    ["sans-serif", "serif"],
    ["upright", "slant"],
    ["rounded", "angular"]
]
def find_font_pair(font_A):
    for pair in font_styles:
        if font_A in pair:
            # Find the other value in the pair
            font_B = pair[1] if font_A == pair[0] else pair[0]
            return font_B
    # If no matching pair is found, return None
    return None
    
new_data_list = []
# Iterate through each element
for element in tqdm(data):
    text_list = element["text"]
    color_list = element.get("color", [])  # If "color" does not exist, return an empty list
    font_list = element.get("font", [])  # If "font" does not exist, return an empty list

    if img_source == "simple":
        ocr_results = element["simple_image_ocr_results"]
        image_path = element["simple_image_path"]
    elif img_source == "enhanced":
        ocr_results = element["enhanced_image_ocr_results"]
        image_path = element["enhanced_image_path"]

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to load image: {image_path}")
        continue

    # Initialize VQA list
    if color_list:
        element["color_VQA"] = []
    elif font_list:
        element["font_VQA"] = []

    for idx, text in enumerate(text_list):
        # Find the OCR result that best matches the current text
        best_match = None
        best_score = 0
        best_coords = None

        try:
            for ocr in ocr_results:
                coords = ocr[0]  # Coordinates in OCR result
                detected_text = ocr[1][0]  # Text in OCR result

                # If the OCR result contains multiple words, try splitting
                detected_words = detected_text.split()  # Split by space
                for word in detected_words:
                    score = similar(text, word)  # Calculate similarity with target text
                    if score > best_score:
                        best_score = score
                        best_match = word
                        best_coords = coords
        except:
            best_score = 0

        text_item = {}
        text_item['matched_text'] = best_match

        # If no matching OCR result is found, fill in as "unknown"
        if not best_match or best_score < float(thresh):  # Increase similarity threshold to 0.6
            print(f"No matching OCR result found for '{text}', filling in as 'unknown'...")
            if color_list:
                text_item['matched_text'] = "NO_TEXT"
            elif font_list:
                text_item['matched_text'] = "NO_TEXT"
            continue

        # Convert coordinates to integers
        best_coords = np.array(best_coords, dtype=np.int32)

        # Get bounding box of the cropped region
        x_min = min(best_coords[:, 0]) - 10
        y_min = min(best_coords[:, 1]) - 10
        x_max = max(best_coords[:, 0]) + 10
        y_max = max(best_coords[:, 1]) + 10

        # Ensure the bounding box is within the image boundaries
        h, w, _ = image.shape
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        # Crop the image
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Save the cropped image
        output_cropped_path = os.path.join(output_folder, f"{element['id']}_{idx}_cropped.png")
        text_item['cropped_image_path'] = output_cropped_path
        cv2.imwrite(output_cropped_path, cropped_image)

        # Process color VQA
        if color_list:
            color_x = color_list[idx] if idx < len(color_list) else "unknown"
            vqa_question = f"The text \"{text}\" is in the color of {color_x}? Answer me using \"yes\" or \"no\"."
            text_item['question'] = vqa_question

        # Process font VQA
        elif font_list:
            font_A = font_list[idx] if idx < len(font_list) else "unknown"
            font_B = find_font_pair(font_A)
            vqa_question = f"The text \"{text}\" is {font_A} or {font_B}? Answer me using either \"{font_A}\" or \"{font_B}\" only."
            text_item['question'] = vqa_question

        # Call GPT API
        with open(output_cropped_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')

        messages = [
            {
                "role": "system",
                "content": "You are an assistant skilled in image data annotation who can accurately identify colors and fonts."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vqa_question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]

        response_data = client.chat.completions.create(
            messages=messages,
            model='gpt-4.1',
            max_tokens=6000,
            # max_retries=5
        )
        # print()

        if response_data:
            content = response_data.choices[0].message.content
            # print(content)
        else:
            content = 'fail to get answer'

        
        print(f"{text} answer:{content}")

        # Update VQA list based on GPT response
        if color_list:
            text_item['answer'] = "Yes" if "yes" in content.lower().strip(".") else "No"
            element['color_VQA'].append(text_item)
        elif font_list:
            text_item['answer'] = "Yes" if font_A.lower() in content.lower().strip(".") else "No"
            element['font_VQA'].append(text_item)

    new_data_list.append(element)


# Save the updated JSON data
with open(output_json, "w") as f:
    json.dump(new_data_list, f, indent=4)

print("Processing completed, results saved!")


path = output_json

data_list = json.load(open(path, "r", encoding="utf-8"))

all_answer_list_color = []
all_answer_list_font = []
all_question_num_color = 0
all_question_num_font = 0
for item in data_list:
    if "color_VQA" in list(item.keys()):
        vqa_item = item['color_VQA']
        all_question_num_color += len(item['text'])
        answer_list = [_['answer'] for _ in vqa_item]
        all_answer_list_color.extend(answer_list)
    elif "font_VQA" in list(item.keys()):
        vqa_item = item['font_VQA']
        all_question_num_font += len(item['text'])
        answer_list = [_['answer'] for _ in vqa_item]
        all_answer_list_font.extend(answer_list)
    # else:
    #     print(item)

right_color = sum([_ == "Yes" for _ in all_answer_list_color])
right_font = sum([_ == "Yes" for _ in all_answer_list_font])

print(f"color: {right_color/all_question_num_color}")
print(f"font: {right_font/all_question_num_font}")


