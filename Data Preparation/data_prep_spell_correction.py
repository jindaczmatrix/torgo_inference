import re
from datasets import load_dataset, DatasetDict, Audio
from huggingsound import SpeechRecognitionModel
from tqdm import tqdm
import json

# Function to remove special characters from text


def remove_special_characters(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

    batch["text"] = re.sub(chars_to_ignore_regex, '',
                           batch["text"]).upper() + " "
    return batch

# Function to save data to a JSON file


def save_to_json(data, filename):
    with open(filename, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)

# Function to preprocess and transcribe the data


def prep_training_data(model, dataset):
    references = []
    for example in tqdm(dataset):
        audio_path = example["audio"]["path"]
        prediction = model.transcribe([audio_path])[0]["transcription"]
        row = {
            "path": audio_path,
            "actual": example["text"].lower(),
            "prediction": prediction,
            "speaker": example["speaker_id"]
        }
        references.append(row)
    return references


def main():
    speaker = 'M04'
    source = "yip-i/torgo_xlsr_finetune-"

    # model to create the transcriptions
    model = SpeechRecognitionModel(source + speaker + "-2")
    data = load_dataset('csv', data_files='output.csv')
    data = data.cast_column("audio", Audio(sampling_rate=16_000))

    # held out speaker data
    speaker_data = data['train'].filter(
        lambda x: x == speaker, input_columns=['speaker_id'])
    
    # other speaker data
    other_speakers_data = data['train'].filter(
        lambda x: x != speaker, input_columns=['speaker_id'])

    # data preprocessing
    speaker_data = speaker_data.map(remove_special_characters)
    other_speakers_data = other_speakers_data.map(
        remove_special_characters)

    # Prepare and transcribe the data for the held-out speaker
    speaker_transcribed = prep_training_data(model, speaker_data)

    # Save the transcribed data for the held-out speaker to a JSON file
    save_to_json(speaker_transcribed, "speaker_M04.json")

    # Prepare and transcribe the data for the remaining speakers
    train_data_transcribed = prep_training_data(model, other_speakers_data)

    # Save the transcribed data for the remaining speakers to a JSON file
    save_to_json(train_data_transcribed, "M04_other_speakers.json")


if __name__ == "__main__":
    main()
