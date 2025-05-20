import whisperx
import json

def audio_to_text_with_diarization(audio_file, auth_token):
    device = "cpu"  
    batch_size = 4  # reduce further if you're low on RAM
    compute_type = "float32"  

    # 1. Transcribe with whisper
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    #print(result["segments"])  # before alignment

    # 2. Align output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    #print(result["segments"])  # after alignment

    # 3. Speaker diarization (you need a Hugging Face token)
    from whisperx.diarize import DiarizationPipeline

    diarize_model = DiarizationPipeline(use_auth_token=auth_token, device=device)

    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    #print(diarize_segments)
    full_converted_list = result["segments"]
    #print(full_converted_list)

    returnlist = []
    for item in full_converted_list:
        returnlist.append([item['text'],item['speaker']])

    with open('output.json', 'w') as file:
        json.dump(returnlist, file) 
        print('Succesfully printed to file')

    return returnlist