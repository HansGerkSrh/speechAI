from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


import json
from scripts import convert_list_to_string

with open('output.json', 'r') as file:
    converted_audio_list = json.load(file)


pre_prompt = "Fasse Folgenes Audio Protokoll als Liste der Inhalte pro SPEAKER zusammen: "
input_text_for_chat_ai = convert_list_to_string(converted_audio_list)


def summarize_text(task_message, text, max_tokens=2048):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained("LeoLM/leo-hessianai-7b")
    model = AutoModelForCausalLM.from_pretrained(
        "LeoLM/leo-hessianai-7b",
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        offload_folder="./offload"
    )

    prompt = f"{task_message}\n{text}\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
    
    summary_ids = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.3,      # More deterministic
        do_sample=True,      # Greedy decoding for stable results
        eos_token_id=tokenizer.eos_token_id,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    if "Zusammenfassung:" in summary:
        summary = summary.split("Zusammenfassung:")[-1].strip()

    return summary


summary = summarize_text(pre_prompt,input_text_for_chat_ai)

print("Summary:", summary)