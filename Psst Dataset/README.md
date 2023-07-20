# PSST Dataset:

This repository contains scripts related to running ASR training to post-processing the ASR transcripts using spelling correction for the PSST dataset. Follow the files below in sequence to understand the whole process.

### General Guidelines:

We are not allowed to share the PSST dataset and permission is needed from the ownwers of the dataset that can be obtained by following the link: https://psst.study/

For our experiments, we downloaded the dataset on the Northeastern University research cluster. We completed all the experiments from ASR finetuning to spelling correction on the cluster.

Plan of action document: https://docs.google.com/document/d/1jchdrh5QjkOeQYj4_F-I4G1JKSLP80RZRGQLlOXERsY/edit?usp=sharing


### Files inside the `Finetuning` folder:
- `preprocess.py`: script to convert PSST dataset's tsv files to csv format to be used for finetuning wav2vec2's xlsr-53 model with the PSST dataset
- `finetuning_xlsr_53_PSST_dataset.ipynb`: script for finetuning wav2vec2's xlsr-53 model with the PSST dataset
- All the ASR models are available on Huggingface under Monideep Chakraborti's repository (https://huggingface.co/monideep2255)

### Files inside the `Evaluation` folder:
- to be added soon

### Files and folders inside the `Spelling Correction` folder:
- `data preparation folder`: folder with all the scripts to generate the ASR transcripts to be used for spelling correction.
    - `PSST_script_generation.ipynb`: script to store ASR transcripts to a json file
    - `update_generated_json.ipynb`: script to add multiple correction pronunciations (references) to the generated json files
- `training and evaluation folder`: folder with all the scripts to train and evaluate the ASR transcripts using spelling correction.
    - `training_spell_correction_script.ipynb`: script to train the BART model with the ASR transcripts
    - `evaluation_spell_correction_script.ipynb`: script to generate inference from the trained BART model. The files located in the evaluation folder were used to evaluate the spelling correction model performance.

- All the Spelling correction models are available on Huggingface under Monideep Chakraborti's repository (https://huggingface.co/monideep2255)
