import json
from audio_to_text import audio_to_text_with_diarization
from scripts import *
#from chatai import summarize_text 
from chatai_CPU import summarize_text_gemma_3_1b

path_to_audio = "Audio/Deutsch_Audio_02.mp3"

path_to_token = ".token" #In File gespeicherter Read Token für huggingface

sys_prompt = "Du bekommst Dialoge zwischen mehreren Person, deren Start durch SPEAKER_XX gekenzeichnet ist. Fasse den Inhalt des Dialogs zusammen"

with open(path_to_token,'r') as f: 
    token = f.read()
f.close()

audio_to_text_with_diarization(path_to_audio,token)

with open('output.json', 'r') as file:
    converted_audio_list = json.load(file)
file.close()

input_text= convert_list_to_string(converted_audio_list)

summary = summarize_text_gemma_3_1b(sys_prompt,input_text)

for i in summary[0][0]["generated_text"]:
    print(i)
    print("")

print(summary[0][0]["generated_text"][2]["content"])