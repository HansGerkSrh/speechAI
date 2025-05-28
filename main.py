import json
from audio_to_text import audio_to_text_with_diarization
from scripts import *
from chatai_CPU import *

"""
Beschreibung des Ablaufs 

Nimmt Audiodatei und gewünschte Textlänge als Input
wandelt in "audio_to_text.py" das Audiofile zu einem Transkript mit Speaker diarization um und speichert die ausgabe liste in output.json  
über die Funktion "convert_list_to_string" in scripts.py wird die Ausgabeliste in einen Menschen lesbaren String für die Textai umgewandelt
"chatai_CPU.py" lässt Lokal Gemma3_1b laufen und fasst das eingegebene Textfile basierend auf den festgelegten systempromt zusammen 
das Ergebniss davon wird an main zurückgegeben und in die Konsole geprinted
"""

def main(path_to_audio,max_new_tokens = 600):
    ### Audio transkription ###

    path_to_token = ".token" #In File gespeicherter Read Token für huggingface

    #lesen des Huggingface read Tokens zum initialen laden der libaries  
    with open(path_to_token,'r') as f: 
        token = f.read()
    f.close()

    #gives to audio file to the transcription ai, which saves the transkript into a .json 
    audio_to_text_with_diarization(path_to_audio,token)


    ### summarising transkript ###

    #Prompt der an das System gegeben wird als "Aufgabe" was mit dem nachfolgenden Text zu tun ist 
    sys_prompt = "Du bekommst Dialoge zwischen mehreren Person, deren Start jeweils durch SPEAKER_XX gekenzeichnet ist. Fasse den Inhalt des Dialogs zusammen"

    #open and read the saved audio log
    with open('output.json', 'r') as file:
        converted_audio_list = json.load(file)
    file.close()

    #converts the json list into a readable String for the text ai 
    input_text= convert_list_to_string(converted_audio_list)

    #prompts the chat ai model (either 1b or 4b) with the sys_prompt and inputs the audio log 
    summary = summarize_text_gemma_3_1b(sys_prompt,input_text,max_new_tokens)

    #output print of the entire output
    for i in summary[0][0]["generated_text"]:
        print(i)
        print("")

    #only prints the summary 
    print(summary[0][0]["generated_text"][2]["content"])



#path_to_audio = "Audio/Deutsch_Audio_02.mp3" #Lokaler Pfad zur Audiodatei
path_to_audio = "Audio/Sprachenlernen.mp3"

#define max lenght of new generated Text
max_new_tokens = 600 

main(path_to_audio,max_new_tokens)