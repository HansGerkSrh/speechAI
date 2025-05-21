Pfad zur Audiodatei in main.py ändern und ausführen 

Setup Guide


Create and activate a virtual environment with venv or uv, a fast Rust-based Python package and project manager.

# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate


Install Transformers in your virtual environment.

# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"


Install Torch 





create .token file 
create offload folder if run on CPU 
create gemma-local folder or named else if its supposed to run offline 
