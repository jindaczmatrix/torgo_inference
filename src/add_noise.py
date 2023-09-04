import numpy as np
import re
import random
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import BallTree
import string

import config as cf


class NoisyDataGenerator:
    def __init__(self):
        self._non_trainable_hyper_params = cf.non_trainable_hyper_params
        self.phonetic_substitutes = cf.phonetic_substitutes
        self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased')
        self.random_seed = self._non_trainable_hyper_params["global_random_seed"]
        self._set_random_seed()

    def _set_random_seed(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    @staticmethod
    def shuffle_data(sentence_df):
        return sentence_df.sample(frac=1.0)

    def generate_context(self, sentence_df):
        data_rows = []
        sentences = self.shuffle_data(sentence_df)["sentences"].values
        sentences = np.random.choice(sentences, size=self._non_trainable_hyper_params["sample_size"], replace=False)
        embeddings = self.sentence_model.encode(sentences,
                                                batch_size=self._non_trainable_hyper_params["emb_batch_size"])
        tree = BallTree(np.array(embeddings), leaf_size=2)

        for i in range(len(embeddings)):
            closest_neighbour_positions = tree.query(embeddings[i:i + 1],
                                                     k=self._non_trainable_hyper_params["k_in_knn"],
                                                     return_distance=False)
            data_rows.append([sentences[i], self.join_and_introduce_spaces(sentences[i]),
                              self.join_and_introduce_spaces(sentences[closest_neighbour_positions[0][1]]),
                              self.join_and_introduce_spaces(sentences[closest_neighbour_positions[0][-1]])])

        return data_rows

    def generate_noise(self, data, words):
        noise_data = []
        if words:
            noise_data = self.add_noise_words(data)
        else:
            data = self.generate_context(data)
            for record in data:
                noise_data.append(self.add_noise_sentences(record))
        return noise_data

    def join_and_introduce_spaces(self, sentence):
        num_spaces = sentence.count(" ")
        sentence = sentence.replace(" ", "").lower()
        positions = []
        while len(positions) < num_spaces:
            rand_num = random.randint(0, len(sentence))
            if rand_num not in (0, len(sentence)) and all(abs(rand_num - num) > 1 for num in positions):
                positions.append(rand_num)
        spaces = [" "] * num_spaces
        new_sentence = list(sentence)

        for i, pos in enumerate(positions):
            new_sentence.insert(pos, spaces[i])

        return self.clean_sentence(''.join(new_sentence))

    @staticmethod
    def clean_sentence(sentence):
        remove_puncts = str.maketrans("", "", string.punctuation)
        return sentence.translate(remove_puncts)

    def add_noise_sentences(self, sim_sents):
        transforms = [self.break_sentence, self.extend_sentence]
        num_transforms = np.random.choice([0, 1, 2], p=[0.15, 0.15, 0.7])
        selected_transforms = np.random.choice(transforms, num_transforms, replace=False, p=[0.6, 0.4])

        for transform in selected_transforms:
            sim_sents = transform(sim_sents)
        noise_sents = self.distort_chars_in_sent(sim_sents)
        return noise_sents

    def break_sentence(self, sim_sents):
        break_percentage = random.uniform(0, self._non_trainable_hyper_params["max_ratio_to_break"])
        chars_to_cut = int(len(sim_sents[0]) * break_percentage)
        strat = random.choice([True, False])

        if strat and 0 < chars_to_cut < len(sim_sents[0]):
            sim_sents[-1] = sim_sents[1][-chars_to_cut:]
            sim_sents[1] = sim_sents[1][:-chars_to_cut]
        if not strat and 0 < chars_to_cut < len(sim_sents[0]):
            sim_sents[2] = sim_sents[1][:chars_to_cut]
            sim_sents[1] = sim_sents[1][chars_to_cut:]

        return sim_sents

    def extend_sentence(self, sim_sents):
        num_words = np.random.choice(self._non_trainable_hyper_params["num_words_to_extend_sent"])
        truncated_candidate = []
        strat = random.choice([True, False])

        if strat:
            space_sep_tokens = sim_sents[2].split()
            if len(space_sep_tokens) > num_words:
                sim_sents[2] = " ".join(space_sep_tokens[-num_words:])
                truncated_candidate = space_sep_tokens[:-num_words]
            else:
                sim_sents[2] = " ".join(space_sep_tokens)
            sim_sents[1] = " ".join([sim_sents[2], sim_sents[1]])
            sim_sents[2] = " ".join(truncated_candidate)
        else:
            space_sep_tokens = sim_sents[-1].split()
            if len(space_sep_tokens) > num_words:
                sim_sents[-1] = " ".join(space_sep_tokens[:num_words])
                truncated_candidate = space_sep_tokens[num_words:]
            else:
                sim_sents[-1] = " ".join(space_sep_tokens)
            sim_sents[-1] = " " + " ".join([sim_sents[-1]])
            sim_sents[1] = " ".join([sim_sents[1], sim_sents[-1]])
            sim_sents[-1] = " ".join(truncated_candidate)

        return sim_sents

    def distort_chars_in_sent(self, sim_sents):
        distorted_sents = []

        for i, sent in enumerate(sim_sents):
            if len(sim_sents) in [0, 1]:
                continue
            if i == 0:
                distorted_sents.append(sent)
                continue

            distorted_sent = self.distort_chars_phonetically(sent)
            distorted_sents.append(distorted_sent)

        return distorted_sents

    def distort_chars_phonetically(self, sentence):
        distorted_sentence = []

        for char in sentence:
            possible_substitute = self.phonetic_substitutes.get(char, char)
            distorted_char = possible_substitute[0] if random.uniform(0, 1) <= self._non_trainable_hyper_params[
                "char_distort_prob"] else char
            distorted_sentence.append(distorted_char)
        return ''.join(distorted_sentence)

    def add_noise_words(self, data):
        sim_words = []
        for word in self.get_words(data):
            phonetics = cf.pronouncing_dict.get(self.remove_special_characters(word.lower()))
            if phonetics:
                phonetics_no_stress = "".join([self.remove_stress(ph) for ph in phonetics[0]])
                sim_words.append([word, phonetics_no_stress.lower()])
        return sim_words



    @staticmethod
    def get_words(data):
        all_words = set()
        for sentence in data['sentences'].sample(n=10000, replace=False, random_state=42):
            doc = cf.nlp(sentence)
            tokens = [token.text.lower() for token in doc if not token.is_punct]
            all_words.update(tokens)

        return list(all_words)

    @staticmethod
    def remove_stress(phoneme):
        return re.sub(r'\d', '', phoneme)

    @staticmethod
    def remove_special_characters(input_string):
        pattern = '[^a-zA-Z0-9]'
        return re.sub(pattern, '', input_string)
