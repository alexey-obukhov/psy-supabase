import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jinja2 import Template
import re

# 1. Load the model and tokenizer
model_name = "rasyosef/Phi-1_5-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 2. Prepare dummy data for the template
context = {
    "user_question": "I feel overwhelmed at work and don't know how to cope.",
    "extracted_topics": ["work stress"],
    "psychological_context": {"emotion": "concern"},
    "conversation_history": [
        {"question": "How can I manage my anxiety?", "answer": "Try to focus on your breathing and take things one step at a time."},
        {"question": "What if I can't sleep?", "answer": "Establish a calming bedtime routine and avoid screens before bed."}
    ]
}

# 3. Load and render the template from a hardcoded path
with open("templates/dynamic_rag_therapy.j2", "r", encoding="utf-8") as f:
    template_content = f.read()
template = Template(template_content)
prompt = template.render(**context)

print("----- PROMPT SENT TO MODEL -----")
print(prompt)
print("----- END PROMPT -----")

# 4. Generate the response
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 5. Extract the assistant's reply
match = re.search(r"<\|im_start\|>assistant\s*(.*?)(<\|im_end\|>|$)", output_text, re.DOTALL)
response = match.group(1).strip() if match else output_text.strip()

print("\n----- ASSISTANT'S REPLY -----")
print(output_text)
print("----- END REPLY -----")