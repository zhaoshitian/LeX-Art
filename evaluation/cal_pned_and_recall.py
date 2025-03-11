import numpy as np
import json
import os
from scipy.optimize import linear_sum_assignment

def normalized_edit_distance(s1, s2):
    """Calculate the normalized edit distance (NED) between two strings."""
    len_s1 = len(s1)
    len_s2 = len(s2)
    max_len = max(len_s1, len_s2)
    if max_len == 0:
        return 0.0
    # Calculate the edit distance
    dp = np.zeros((len_s1 + 1, len_s2 + 1))
    for i in range(len_s1 + 1):
        for j in range(len_s2 + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1,      # Deletion
                               dp[i][j-1] + 1,      # Insertion
                               dp[i-1][j-1] + cost) # Substitution
    # Normalize the edit distance
    ned = dp[len_s1][len_s2] / max_len
    return ned

def calculate_recall(list1, list2, threshold=0.3):
    """
    Calculate the recall of list2 with respect to list1.
    Ensure that each element in list1 and list2 can only be used once.
    """
    true_positives = 0
    total = len(list1)
    used_indices_list1 = set()  # Record the indices of matched elements in list1
    used_indices_list2 = set()  # Record the indices of matched elements in list2

    for i, gt in enumerate(list1):
        if i in used_indices_list1:
            continue  # If the element in list1 has already been matched, skip
        for j, pred in enumerate(list2):
            if j in used_indices_list2:
                continue  # If the element in list2 has already been matched, skip
            ned = normalized_edit_distance(gt, pred)
            if ned <= threshold:
                true_positives += 1
                used_indices_list1.add(i)  # Mark the element in list1 as used
                used_indices_list2.add(j)  # Mark the element in list2 as used
                break  # Break the inner loop after a successful match
    
    recall = true_positives / total
    return recall

def matching_based_nled(test_list, gt_list):
    # Calculate the matching-based normalized edit distance
    len_test, len_gt = len(test_list), len(gt_list)
    
    # Build the cost matrix
    cost_matrix = np.zeros((len_test, len_gt))
    for i, test_item in enumerate(test_list):
        for j, gt_item in enumerate(gt_list):
            cost_matrix[i][j] = normalized_edit_distance(test_item, gt_item)
    
    # Use the Hungarian algorithm to find the optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Calculate the matched cost and unmatched cost
    matched_cost = cost_matrix[row_ind, col_ind].sum()
    unmatched_cost = abs(len_test - len_gt)  # Number of unmatched elements
    
    # Total cost
    total_cost = matched_cost + unmatched_cost
    
    # Normalize
    max_len = max(len_test, len_gt)
    return total_cost 

def parse_text(simple_caption):
    text = simple_caption.split("\"", -1)[-2].lower()
    return [text]

def collect_data(path):
    try:
        data_list = json.load(open(path, "r", encoding="utf-8"))
        ocr_results_simple_image = [item['simple_image_ocr_results'] for item in data_list]
        ocr_results_enhanced_image = [item['enhanced_image_ocr_results'] for item in data_list]
        # Process the OCR results for simple_image
        ocr_results_simple_image = [
            [item[1][0].lower() for item in ocr_list] if ocr_list else [""]
            for ocr_list in ocr_results_simple_image
        ]

        # Process the OCR results for enhanced_image
        ocr_results_enhanced_image = [
            [item[1][0].lower() for item in ocr_list] if ocr_list else [""]
            for ocr_list in ocr_results_enhanced_image
        ]
        if "simplebench" in path or "createbench" in path or "anytext" in path:
            gt_text_list = [parse_text(item['caption']) for item in data_list]
        else:
            gt_text_list = [[_.lower() for _ in item['text']] for item in data_list]

        return ocr_results_simple_image, ocr_results_enhanced_image, gt_text_list
    except:
        return [], [], []

if __name__ == "__main__":
    path = os.getenv("BENCH_RESULT_PATH")

    all_nled_simple_image, all_nled_enhanced_image = [], []
    all_recall_simple_image, all_recall_enhanced_image = [], []

    ocr_results_simple_image, ocr_results_enhanced_image, gt_text_list = collect_data(path)
    
    # Calculate NLED
    nled_list_simple_image = [matching_based_nled(A, B) for A, B in zip(gt_text_list, ocr_results_simple_image)]
    nled_list_enhanced_image = [matching_based_nled(A, B) for A, B in zip(gt_text_list, ocr_results_enhanced_image)]

    # Calculate Recall
    recall_list_simple_image = [calculate_recall(A, B) for A, B in zip(gt_text_list, ocr_results_simple_image)]
    recall_list_enhanced_image = [calculate_recall(A, B) for A, B in zip(gt_text_list, ocr_results_enhanced_image)]

    # Aggregate results
    all_nled_simple_image += nled_list_simple_image
    all_nled_enhanced_image += nled_list_enhanced_image
    all_recall_simple_image += recall_list_simple_image
    all_recall_enhanced_image += recall_list_enhanced_image

    # Output results
    print(f"NLED score (simple): {sum(all_nled_simple_image)/len(all_nled_simple_image)}")
    print(f"NLED score (enhanced): {sum(all_nled_enhanced_image)/len(all_nled_enhanced_image)}")
    print(f"Recall score (simple): {sum(all_recall_simple_image)/len(all_recall_simple_image)}")
    print(f"Recall score (enhanced): {sum(all_recall_enhanced_image)/len(all_recall_enhanced_image)}")