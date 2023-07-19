# Torgo_Testing/Torgo Inference:

This repository contains files related to running ASR training to post-processing the ASR transcripts using spelling correction. Follow the files below to understand the whole process.

### **General Guidelines:**

- Torgo files are expected base on output.csv to be in content/downloads/Torgo.
- All Torgo speakers are directly under the Torgo folder.
- Testing works best in .py file rather than Notebook.

### Automatic Speech Recognition Training:
1. Basic understanding of wav2vec2 (will need to have a Huggingface and Google Collab account):
   - `Finetuning wav2vec tutorial`: https://huggingface.co/blog/fine-tune-wav2vec2-english
2. Work with Torgo dataset:
   - ASR finetuning of Torgo dataset with xlsr-53 wav2vec2 model: https://colab.research.google.com/drive/1kX_pBURiaujpuDYaB8O1hrLLQrK8bWsQ
3. Adding a language model: https://colab.research.google.com/drive/1AIgP6lc7BZTDrlU5yu83R05QDesWv4mw
4. All the ASR models are available on Huggingface under Ian Yip's repository (https://huggingface.co/yip-i)

### Files inside the `Data Preparation` folder:
- `asr_testing_jonatas.ipynb`: Collab script with ASR testing with model available in research community (jonatasgrosman)
- `asr_testing_jonatas.py`: Python script for the `asr_testing_jonatas.ipynb` file
- `asr_testing_lm.py`: Python script with ASR testing with language model
- `correction_algo_prep.py and correction_data_demo`: base data preparation file and demo file for `data_prep_spell_correction.py` file
- `data_prep_spell_correction.py`: data preparation for spelling correction
- `output_og.csv and output.csv`: datasets for ASR training

### **Running the script to prepare data for spelling correction using machine translation techniques:**

1. All files can be found inside the `Data Preparation` folder.

2. Create a virtual environment and install the requirements.txt file for all the libraries. Please make sure that the Python version is between 3.7 to 3.10. Some libraries are not compatible yet with the higher versions.

3. Inside the virtual environment, run the file `data_prep_spell_correction.py`.

4. The way the code is setup, the Torgo audio files are accessed from content/downloads/Torgo. Please make sure to download the Torgo speakers locally or mount in Google Drive if running on Google Collab.

   - Need to create a folder called content to store the Torgo files

5. There are two types of files:
   - `speaker_ID.json` represents the transcripts for the specific speaker
   - `ID_other_speakers.json` represents the transcripts for all the speakers except for the speaker ID mentioned in the file name.

### **Spelling correction using machine translation techniques:**

1. The script for machine translation can be found in the `Machine Translation` folder

   - Plan of action: https://docs.google.com/document/d/1aAzRwka9W3B8xB6kuggZy0P93VlbCZfjxQraok2Vl10/edit?usp=sharing

2. Files inside the folder:
   - `Machine_Translation_F01.ipynb`: script with all the training, evaluation and testing code for speaker F01
   - `Spell_Correction_Training_Script_Machine_Translation.ipynb`: script for training the machine translation model
   - `Spell_Correction_Evaluation_and_Testing_Script_Machine_Translation.ipynb`: script for testing the machine translation model
   - `machine_translation_tutorial.ipynb`: script for machine translation tutorial from HuggingFace

3. Links:
   -  Training Script: https://colab.research.google.com/drive/13OUCdq2V_IpmMvF_NOwZVwMZSAh0GfYy?usp=sharing
   -  Evaluation and Testing Script: updated: evaluation and testing.ipynb   
   -  Script for F01 Speaker : https://colab.research.google.com/drive/1lhIKDb2JITULf3bgC97YQ1XM8frPV-AJ?usp=sharing

4. All the Spelling correction models are available on Huggingface under Monideep Chakraborti's repository (https://huggingface.co/monideep2255)


