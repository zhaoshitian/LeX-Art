# Evaluation Data Format

To calculate the score of your generated images on benchamrks, you need to format your evaluation data into the below format:
```json
[
    {
        "image_name": "the/path/to/the/generated/image",
        "caption": "caption",
        "refined_caption": "the/refined/caption"
    },
    ......
]
```

## Calculate Aesthetic Score
```shell
export BENCH_RESULT_PATH=/path/to/the/result/files
export SAVE_PATH=/path/to/save/the/aesthetic/score
python cal_aesthetic.py
```

## Calculate Color and Font Score
```shell
export THRESH=0.3
export IMG_SOURCE="simple" # "simple" or "enhanced"
export MODEL_NAME="flux"
export BENCH_RESULT_PATH=/path/to/the/result/files
export SAVE_PATH=/path/to/save/the/color/and/font/score
python cal_color_and_font_score.py
```

## Calculate PNED and Recall Score
```shell
export BENCH_RESULT_PATH=/mnt/petrelfs/zhaoshitian/Flux-Text/results_from_anytext/anytext_with_ocr.json
export SAVE_PATH=/mnt/petrelfs/zhaoshitian/Flux-Text/benchmarks/final_result_files_with_qa_score/anytext_with_ocr.json
python cal_pned_and_recall.py
```

## Calculate Position Score
```shell
export IMG_SOURCE="simple" # "simple" or "enhanced"
export BENCH_RESULT_PATH=/mnt/petrelfs/zhaoshitian/Flux-Text/results_from_anytext/anytext_with_ocr.json
export SAVE_PATH=/mnt/petrelfs/zhaoshitian/Flux-Text/benchmarks/final_result_files_with_qa_score/anytext_with_ocr.json
python cal_position_score.py
```