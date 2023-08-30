The Automatic Speech Recognition (ASR) outputs for impaired speakers are not of satisfactory quality, primarily due to a high occurrence of spelling errors. To improve the accuracy of ASR outputs for impaired speakers, we aim to develop a model that can effectively correct these spelling errors. Our approach involves augmenting the dataset by introducing noise, and then training a machine learning model to perform error corrections on ASR output. The ultimate goal is to enhance the overall performance of ASR systems for impaired speakers, leading to more accurate and reliable transcriptions of their speech.

## Dataset
For V0 we will adopt a two-fold approach: Firstly, we will focus on augmenting the Tatoeba dataset, which contains diverse and multilingual sentences. To introduce noise, we will leverage word embeddings and randomly modify the input sentences while preserving their semantic meaning. By doing so, we aim to create a more extensive and varied dataset that can better represent real-world ASR challenges. Secondly, we will combine a sample of the augmented Tatoeba dataset with the ASR outputs from the Torgo dataset. By merging these noisy sentences with the original ASR outputs, we aim to create a more robust and generalizable dataset. 
Finally, With this enriched dataset, we plan to finetune a pre-trained model like BART for ASR error correction.

## Implementation Details

- Load the Tatoeba dataset from the specified file path using pandas.
- Create an instance of the SentenceTransformer model `distiluse-base-multilingual-cased` to compute sentence embeddings.
- The `_set_random_seed()` method generates a random seed for reproducibility
- The `shuffle_data()` method to shuffle the sentence DataFrame.
- Shuffle the data and select a random sample of sentences.
- Compute sentence embeddings using the SentenceTransformer model.
- Use BallTree to find the `3` nearest neighboring sentences for each sentence.
- Implement the `join_and_introduce_spaces()` method to remove spaces, lowercase the sentence, and introduce spaces at random positions (the randomness is controlled so they are not leading nor trailing nor continous spaces).
- Two primary transformations: `break_sentence` and `extend_sentence` are selected randomly based on predefined probabilities. The number of transformations is also randomly determined to maintain variability.
- Randomly truncates a portion of first alternative based on a randomly chosen break percentage, and appends or prefixes the truncated portion to either second or third alternative, respectively, based on a randomly chosen strategy.
- Modifies first alternative by extending it with words either from second or third alternative based on a randomly chosen strategy and number of words
- Define a dictionary `phonetic_substitutes` to hold phonetically similar characters for distortion and apply phonetic character-level distortion to sentences using this dictionary

## Run
Generate noise data for the configured sample size (if required modify non-trainable hyperparameters in the config file) using:
```commandline
python main.py
```
