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


all_answer_list_position = []
all_question_num_position = 0

json_path = f"/mnt/petrelfs/zhaoshitian/Flux-Text/benchmarks/final_result_files_with_qa_score_and_ocr/final_easy_seed42_expfluxtext-10k-e5.json"
img_source = "simple"


def check_box_position(position, box_coords, image_size=1024):
    x1, y1, x2, y2 = box_coords
    
    # Calculate the center of the box
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2
    
    # Calculate the center of the image
    image_center_x = image_size / 2
    image_center_y = image_size / 2
    
    # Define the tolerance for the position check
    tolerance = 50  # You can adjust this value based on how strict you want the check to be
    
    if position == "lower right corner":
        # Check if the box center is in the lower right quadrant
        return "yes" if (box_center_x >= image_center_x + tolerance and box_center_y >= image_center_y + tolerance) else "no"
    
    elif position == "lower left corner":
        # Check if the box center is in the lower left quadrant
        return "yes" if (box_center_x <= image_center_x - tolerance and box_center_y >= image_center_y + tolerance) else "no"
    
    elif position == "bottom":
        # Check if the box center is in the bottom half
        return "yes" if (box_center_y >= image_center_y + tolerance) else "no"
    
    elif position == "left":
        # Check if the box center is in the left half
        return "yes" if (box_center_x <= image_center_x - tolerance) else "no"
    
    elif position == "upper left corner":
        # Check if the box center is in the upper left quadrant
        return "yes" if (box_center_x <= image_center_x - tolerance and box_center_y <= image_center_y - tolerance) else "no"
    
    elif position == "upper right corner":
        # Check if the box center is in the upper right quadrant
        return "yes" if (box_center_x >= image_center_x + tolerance and box_center_y <= image_center_y - tolerance) else "no"
    
    elif position == "right":
        # Check if the box center is in the right half
        return "yes" if (box_center_x >= image_center_x + tolerance) else "no"
    
    elif position == "top":
        # Check if the box center is in the top half
        return "yes" if (box_center_y <= image_center_y - tolerance) else "no"
    
    elif position == "center":
        # Check if the box center is near the image center
        return "yes" if (abs(box_center_x - image_center_x) <= tolerance and abs(box_center_y - image_center_y) <= tolerance) else "no"
    
    else:
        return "no"

# Define a function to calculate string similarity
def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Load JSON data
with open(json_path, "r") as f:  # Replace with your JSON file path
    data = json.load(f)

position_list = [
    "lower right corner",
    "lower left corner",
    "bottom",
    "left",
    "upper left corner",
    "upper right corner",
    "right",
    "top",
    "center"
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
    position_list = element.get("position", [])  # If "font" does not exist, return an empty list

    if img_source == "simple":
        ocr_results = element["simple_image_ocr_results"]
    elif img_source == "enhanced":
        ocr_results = element["enhanced_image_ocr_results"]

    # Initialize VQA list
    if color_list:
        element["color_VQA"] = []
    elif font_list:
        element["font_VQA"] = []
    elif position_list:
        element["position_answer"] = []
        all_question_num_position += len(element['text'])

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
        if not best_match or best_score < 0.6:  # Increase similarity threshold to 0.6
            print(f"No matching OCR result found for '{text}', filling in as 'unknown'...")
            if color_list:
                text_item['matched_text'] = "NO_TEXT"
            elif font_list:
                text_item['matched_text'] = "NO_TEXT"
            continue

        # Convert coordinates to integers
        best_coords = np.array(best_coords, dtype=np.int32)

        # Get the bounding box of the cropped region
        x_min = min(best_coords[:, 0])
        y_min = min(best_coords[:, 1])
        x_max = max(best_coords[:, 0])
        y_max = max(best_coords[:, 1])

        # Ensure the bounding box is within the image boundaries
        h, w = 1024, 1024
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        if position_list:
            position_x = position_list[idx]
            box_coords = x_min, y_min, x_max, y_max
            position_answer = check_box_position(position_x, box_coords, image_size=1024)
            text_item['position'] = position_x
            text_item['position_answer'] = position_answer

            all_answer_list_position.append(position_answer)


            print(text_item)

    new_data_list.append(element)

# Save the updated JSON data
# with open(output_json, "w") as f:
#     json.dump(new_data_list, f, indent=4)

# print("Processing completed, results saved!")

print(f"acc: {sum([_ == 'yes' for _ in all_answer_list_position])/all_question_num_position}")