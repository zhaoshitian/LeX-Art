### Prepare data
Download from [huggingface](https://huggingface.co/datasets/X-ART/LeX-R1-60K).

### Inference with transformers
```python
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

SYSTEM_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> <answer> answer here </answer>."
)

model_path = 'X-ART/LeX-Enhancer-full'
simple_caption = "A thank you card with the words very much, with the text on it: \"VERY\" in black, \"MUCH\" in yellow."

def create_chat_template(user_prompt):
    return [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "<think>"}
    ]

def create_direct_template(user_prompt):
    return user_prompt + "<think>" # better

def create_user_prompt(simple_caption):
    return (
        "Below is the simple caption of an image with text. Please deduce the detailed description of the image based on this simple caption. "
        "Note: 1. The description should only include visual elements and should not contain any extended meanings. "
        "2. The visual elements should be as rich as possible, such as the main objects in the image, their respective attributes, "
        "the spatial relationships between the objects, lighting and shadows, color style, any text in the image and its style, etc. "
        "3. The output description should be a single paragraph and should not be structured. "
        "4. The description should avoid certain situations, such as pure white or black backgrounds, blurry text, excessive rendering of text, "
        "or harsh visual styles. "
        "5. The detailed caption should be human readable and fluent. "
        "6. Avoid using vague expressions such as \"may be\" or \"might be\"; the generated caption must be in a definitive, narrative tone. "
        "7. Do not use negative sentence structures, such as \"there is nothing in the image,\" etc. The entire caption should directly describe the content of the image. "
        "8. The entire output should be limited to 200 words.\n\n"
        "SIMPLE CAPTION: {0}"
    ).format(simple_caption)
    
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

# Tokenize the input prompt
messages = create_direct_template(create_user_prompt(simple_caption))  # 3.for direct template
input_ids = tokenizer.encode(messages, return_tensors="pt")

# Generate text using the model
streamer = TextStreamer(tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True)
output = model.generate(
    input_ids.to(model.device), 
    max_length=2048,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.6,
    repetition_penalty=1.1,
    streamer=streamer
)

# Print the generated text
print("*" * 80)
# print(generated_text)
```
