import csv
import numpy as np
import pandas as pd
import Levenshtein as lev
import config as cf
from src.add_noise import NoisyDataGenerator


def select_best_alternative(row):
    alternatives = [f"Alternative{i}" for i in range(1, 4)]

    scores = []
    for alt in alternatives:
        length_difference = abs(len(row['Sentence']) - len(row[alt]))
        edit_dist = lev.distance(row['Sentence'], row[alt])
        penalty = np.exp(-length_difference)
        score = penalty * edit_dist
        scores.append((alt, score))

    best_alternative = sorted(scores, key=lambda x: x[1], reverse=True)[0]
    return best_alternative[0]


def run(words=False):
    data_generator = NoisyDataGenerator()

    data_file = "en_sentences.tsv"
    data_file_path = r"/data/tatoeba/en/"
    td_data_file_path = f"{cf.proj_base_path}{data_file_path}{data_file}"
    tb_data = pd.DataFrame(columns=["sentences"])
    tb_data["sentences"] = pd.read_csv(td_data_file_path,
                                       encoding="utf-8",
                                       delimiter='\t',
                                       quoting=csv.QUOTE_NONE).iloc[:, -1]
    noise_batch = data_generator.generate_noise(tb_data, words)
    if words:
        noise_df = pd.DataFrame(noise_batch, columns=['word', 'phonetic_alterntive'])
        noise_df.to_csv(f"{cf.proj_base_path}{data_file_path}words_noise_data_sample.csv", index=False)
    else:
        noise_df = pd.DataFrame(noise_batch, columns=['Sentence', 'Alternative1', 'Alternative2', 'Alternative3'])
        noise_df['Best_Alternative'] = noise_df.apply(select_best_alternative, axis=1)
        noise_df.to_csv(f"{cf.proj_base_path}{data_file_path}sentences_noise_data_sample.csv", index=False)


if __name__ == "__main__":
    run(words=True)
