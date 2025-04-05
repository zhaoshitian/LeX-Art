# Evaluation Data Format

To calculate the score of your generated images on benchamrks, you need to format your evaluation data into the below format:
```shell
[
    {
        "image_name": "image/id/or/name",
        "caption": "caption",
        "enhanced_caption": "the/enhanced/caption",
        "text": "the/ground/truth/text", # if LeX-Bench or none
        # "color" or "font" or "position" if LeX-Bench or none
        "simple_image_path" : "the/path/to/the/generated/image/using/simple/caption",
        "enhanced_image_path" : "the/path/to/the/generated/image/using/enhanced/caption",
        "simple_image_ocr_results": "ocr/results/from/paddle-ocr-v3",
        "enhanced_image_ocr_results": "ocr/results/from/paddle-ocr-v3", 
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
export BENCH_RESULT_PATH=/path/to/the/result/files
export SAVE_PATH=/path/to/save/the/PNED/and/Recall/score
python cal_pned_and_recall.py
```

## Calculate Position Score
```shell
export IMG_SOURCE="simple" # "simple" or "enhanced"
export BENCH_RESULT_PATH=/path/to/the/result/files
export SAVE_PATH=/path/to/save/the/position/score
python cal_position_score.py
```
