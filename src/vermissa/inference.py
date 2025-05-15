from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load the fine-tuned model (from the output directory)
fine_tuned_model = AutoModelForCausalLM.from_pretrained("../../models/model=gpt2/book=sherlock/checkpoint-516")

# Load the original GPT-2 model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "Holmes leaned back in his chair and tapped the pipe against his palm. 'This pipe,' he said, 'belonged to none other than"
inputs = tokenizer(prompt, return_tensors="pt")

fine_outputs = fine_tuned_model.generate(
    **inputs,
    max_new_tokens=50,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
    repetition_penalty=1.2,
)
print("Fine-tuned GPT-2:", tokenizer.decode(fine_outputs[0], skip_special_tokens=True))


print("")
# Original
base_outputs = base_model.generate(
    **inputs,
    max_new_tokens=50,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
    repetition_penalty=1.2,
)
print("Original GPT-2:", tokenizer.decode(base_outputs[0], skip_special_tokens=True))