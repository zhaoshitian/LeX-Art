from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "X-ART/LeX-Enhancer_FULL"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """
Blow is the simple caption of an image with text. Please deduce the detailed description of the image based on this simple caption. Note: 1. The description should only include visual elements and should not contain any extended meanings. 2. The visual elements should be as rich as possible, such as the main objects in the image, their respective attributes, the spatial relationships between the objects, lighting and shadows, color style, any text in the image and its style, etc. 3. The output description should be a single paragraph and should not be structured. 4. The description should avoid certain situations, such as pure white or black backgrounds, blurry text, excessive rendering of text, or harsh visual styles. 5. The detailed caption should be human readable and fluent. 6. Avoid using vague expressions such as "may be" or "might be"; the generated caption must be in a definitive, narrative tone. 7. Do not use negative sentence structures, such as "there is nothing in the image," etc. The entire caption should directly describe the content of the image. 8. The entire output should be limited to 200 words.
SIMPLE CAPTION: A white background with the top of the bertrand logo and the bottom of the logo, with text on it: "TOP", "B", "THE", "BERTRAND", "SUSTLIROS".
"""
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
