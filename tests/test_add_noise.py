import unittest
import random
import string
from unittest.mock import MagicMock
from src.add_noise import NoisyDataGenerator


class TestNoisyDataGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = NoisyDataGenerator()

    def test_shuffle_data(self):
        sentence_df = MagicMock()
        sentence_df.sample.return_value = MagicMock(values=["sent1", "sent2", "sent3"])

        shuffled_data = self.generator.shuffle_data(sentence_df)
        sentence_df.sample.assert_called_with(frac=1.0)
        self.assertEqual(shuffled_data, sentence_df.sample.return_value)

    def test_join_and_introduce_spaces(self):
        sentence = "This is a test sentence"
        noisy_sentence = self.generator.join_and_introduce_spaces(sentence)
        self.assertEqual(len(noisy_sentence), len(sentence) + sentence.count(" "))
        self.assertFalse(noisy_sentence.startswith(" "))
        self.assertFalse(noisy_sentence.endswith(" "))
        self.assertTrue(all(a != " " and b != " " for a, b in zip(noisy_sentence, noisy_sentence[1:])))

    def test_clean_sentence(self):
        dirty_sentence = "Hello, world! How are you?"
        cleaned_sentence = self.generator.clean_sentence(dirty_sentence)
        self.assertEqual(cleaned_sentence, "Hello world How are you")

    def test_distort_chars_phonetically(self):
        sentence = "hello"
        distorted_sentence = self.generator.distort_chars_phonetically(sentence)
        self.assertEqual(len(distorted_sentence), len(sentence))
        self.assertTrue(all(char != distorted_char for char, distorted_char in zip(sentence, distorted_sentence)))

    def test_break_sentence(self):
        sim_sents = ["Original sentence", "Modified sentence", "Extended sentence"]
        new_sents = self.generator.break_sentence(sim_sents)
        self.assertNotEqual(sim_sents, new_sents)

    def test_extend_sentence(self):
        sim_sents = ["Original sentence", "Modified sentence", "Extended sentence"]
        new_sents = self.generator.extend_sentence(sim_sents)
        self.assertNotEqual(sim_sents, new_sents)

    def test_generate_noise(self):
        data = [["sentence1", "modified1", "extended1", "extended2"], ["sentence2", "modified2", "extended3", "extended4"]]
        noise_data = self.generator.generate_noise(data)
        self.assertEqual(len(noise_data), len(data))

    def test_generate_context(self):
        sentence_df = MagicMock()
        sentence_df.__getitem__.return_value = MagicMock(values=["sentence1", "sentence2"])
        context_data = self.generator.generate_context(sentence_df)
        self.assertEqual(len(context_data), len(sentence_df.__getitem__.return_value))


if __name__ == '__main__':
    unittest.main()
