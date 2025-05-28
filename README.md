
## Readme ist noch nicht fertig und muss noch verbessert werden eine Detailierte beschreibung des Ablaufs ist in main.py

Pfad zur Audiodatei in main.py ändern und ausführen 

Setup Guide


-Create and activate a virtual environment with venv or uv

# venv
python -m venv .my-env
source .my-env/bin/activate

-Install Transformers in your virtual environment.

# pip
pip install "transformers[torch]"


- Install Torch 

- pip install whisperx



-create ".token" file  ----- #put huggingface read token inside and request acces to:

     https://huggingface.co/pyannote/segmentation-3.0 and https://huggingface.co/pyannote/speaker-diarization-3.1
     you can create the read token here: https://huggingface.co/security-checkup?cookieId=e8bbda46-f7cc-4656-a68f-bab7f724e21c

-create "offload" folder if run on CPU 
(currently only works this way)


