import os

abs_path = os.path.abspath(__file__)
proj_base_path = os.path.dirname(abs_path)

#
non_trainable_hyper_params = {
    "max_ratio_to_break": 0.4,
    "max_ratio_to_extend": 0.4,
    "global_random_seed": 300,
    "sample_size": 10000,
    "emb_batch_size": 128,
    "k_in_knn": 3,
    "num_words_to_extend_sent": list(range(1, 11)),
    "char_distort_prob": 0.3
}

# Currently it is a basic dictionary and might not cover all possible phonetic variations
# TODO add n-gram subsitutes and m possibilities (in reference to character level not tokens). For example, {'ie': ['ee', 'ae'],'ear': ['eer', 'year']}
phonetic_substitutes = {
    'b': ['p'],
    'd': ['t'],
    't': ['d'],
    'p': ['b'],
    'k': ['g'],
    'g': ['k'],
    'f': ['v'],
    'v': ['f'],
    's': ['z'],
    'z': ['s'],
    'm': ['n'],
    'n': ['m'],
    'l': ['r'],
    'r': ['l']
}