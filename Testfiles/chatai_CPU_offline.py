### Testing input ###

# import json
# from scripts import convert_list_to_string

# with open('output.json', 'r') as file:
#     converted_audio_list = json.load(file)

# sysprompt = "Du bekommst Dialoge zwischen mehreren Personen, deren Start durch SPEAKER_XX gekennzeichnet ist. Fasse den Inhalt des Dialogs zusammen."
# input_text_for_chat_ai = convert_list_to_string(converted_audio_list)
# max_new_tokens = 500


def summarize_text_gemma_3_1b(sysprompt, inputtext, max_new_tokens):
    from transformers import AutoTokenizer, Gemma3ForCausalLM
    import torch

    model_id = "google/gemma-3-1b-it"

    ### First Time Installation###
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = Gemma3ForCausalLM.from_pretrained(model_id).eval().to("cpu")

    # tokenizer.save_pretrained("./gemma-local")
    # model.save_pretrained("./gemma-local")

    tokenizer = AutoTokenizer.from_pretrained("./gemma-local", local_files_only=True)
    model = Gemma3ForCausalLM.from_pretrained("./gemma-local", local_files_only=True)

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": sysprompt}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": inputtext}]
            },
        ]
    ]

    # Prepare input
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():   
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_output


def chatbot_output_cleanup(decoded_output):
    import re

    output = re.split(r'\nmodel\n', decoded_output[0], maxsplit=1)

    if len(output) > 1:
        output = output[1]
    else:
        output = "chatbot didnt produce any response"

    return output
    

###output for testing###

# raw_output = summarize_text_gemma_3_1b(sysprompt,input_text_for_chat_ai,max_new_tokens)
# output = chatbot_output_cleanup(raw_output)
# print(output)