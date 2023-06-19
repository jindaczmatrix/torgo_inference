# Torgo_Testing/Torgo Inference:

### **General Guidelines:**

- Torgo files are expected base on output.csv to be in content/downloads/Torgo.
- All Torgo speakers are directly under the Torgo folder.
- Testing works best in .py file rather than Notebook.

### **Running the script to prepare data for spelling corrrection using machine translation techniques:**

1. All files can be found inside the `Data Preparation` folder.

2. Create a virtual environment and install the requirements.txt file for all the libraries. Please make sure that the Python version is between 3.7 to 3.10. Some libraries are not compatible yet with the higher versions.

3. Inside the virtual environment, run the file `data_prep_spell_correction.py`.

4. The way the code is setup, the Torgo audio files are accessed from content/downloads/Torgo. Please make sure to download the Torgo speakers locally or mount in Google Drive if running on Google Collab.

   - Need to create a folder called content to store the Torgo files

5. There are two types of files:
   - `speaker_ID.json` represents the transcripts for the specific speaker
   - `ID_other_speakers.json` represents the transcripts for all the speakers except for the speaker ID mentioned in the file name.

### **Spelling corrrection using machine translation techniques:**

1. The script for machine translation can be found in the `Machine Translation` folder

   - Plan of action: https://docs.google.com/document/d/1aAzRwka9W3B8xB6kuggZy0P93VlbCZfjxQraok2Vl10/edit?usp=sharing

2. Files inside the folder:
   - `Machine_Translation_F01.ipynb`: script with all the training, evaluation and testing code for speaker F01
   - `Spell_Correction_Training_Script_Machine_Translation.ipynb`: script for training the machine translation model
   - `Spell_Correction_Evaluation_and_Testing_Script_Machine_Translation.ipynb`: script for testing the machine translation model
   - `machine_translation_tutorial.ipynb`: script for machine translation tutorial from HuggingFace

3. Links:
   -  Training Script : https://colab.research.google.com/drive/13OUCdq2V_IpmMvF_NOwZVwMZSAh0GfYy?usp=sharing
   -  Evaluation and Testing Script : https://colab.research.google.com/drive/1q3QPmm49yJFunvUYI5JFQDddsHIoVzHx?usp=sharing   
   -  Script for F01 Speaker : https://colab.research.google.com/drive/1lhIKDb2JITULf3bgC97YQ1XM8frPV-AJ?usp=sharing


