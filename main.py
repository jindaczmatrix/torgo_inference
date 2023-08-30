import csv
import pandas as pd
import config as cf
from src.add_noise import NoisyDataGenerator


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
    noise_df = pd.DataFrame(noise_batch, columns=['Sentence', 'Alternative1', 'Alternative2', 'Alternative3'])
    noise_df.to_csv(f"{cf.proj_base_path}{data_file_path}noise_data_sample.csv", index=False)


if __name__ == "__main__":
    run(words=True)
