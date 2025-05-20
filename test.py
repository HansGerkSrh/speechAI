from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import json
from scripts import convert_list_to_string

with open('output.json', 'r') as file:
    converted_audio_list = json.load(file)

sysprompt = "Du bekommst Dialoge zwischen mehreren Person, deren Start durch SPEAKER_XX gekenzeichnet ist. Fasse den Inhalt des Dialogs zusammen"
input_text_for_chat_ai = convert_list_to_string(converted_audio_list)



def summarize_text_cpu(sysprompt,inputtext):

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it",
        device_map="auto",               # Automatically place layers on GPU/CPU/disk
        offload_folder="./offload",      # Folder to swap model parts to disk
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": sysprompt },]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": inputtext},]
            },
        ],
    ]

    output = pipe(messages, max_new_tokens=500)
    return output

output = summarize_text_cpu(sysprompt,input_text_for_chat_ai)

for i in output[0][0]["generated_text"]:
    print(i)
    print("")

print(output[0][0]["generated_text"][2]["content"])