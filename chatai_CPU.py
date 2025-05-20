###lokal testing###
import json
from scripts import convert_list_to_string

with open('output.json', 'r') as file:
    converted_audio_list = json.load(file)

sysprompt = "Du bekommst Dialoge zwischen mehreren Person, deren Start durch SPEAKER_XX gekenzeichnet ist. Fasse den Inhalt des Dialogs zusammen"
input_text_for_chat_ai = convert_list_to_string(converted_audio_list)
max_new_tokens = 500


def summarize_text_gemma_3_1b(sysprompt,inputtext,max_new_tokens):
    from transformers import pipeline

    pipe = pipeline("text-generation", model="google/gemma-3-1b-it")

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": sysprompt},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": inputtext},]
            },
        ],
    ]

    output = pipe(messages, max_new_tokens=max_new_tokens)
    return output


def summarize_text_gemma_3_4b(sysprompt,inputtext,max_new_tokens):
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch


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

    output = pipe(messages, max_new_tokens=max_new_tokens)
    return output


###output for testing###

output = summarize_text_gemma_3_4b(sysprompt,input_text_for_chat_ai,max_new_tokens)

for i in output[0][0]["generated_text"]:
    print(i)
    print("")

print(output[0][0]["generated_text"][2]["content"])