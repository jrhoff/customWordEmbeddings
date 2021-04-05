from string import punctuation as PUNCT
PARENT_CODE_LENGTH = 3
SUBCATEGORY_LENGTH = 4

CODE2DESC_PATH = 'code2desc.pickle'
WORD2DF_PATH = 'word2df.pickle'
WORD2TF_PATH = 'word2tf.pickle'
FAMILY2TF_PATH = 'family2tf.pickle'
WORD_EMBEDDING_PATH = 'word_vectors'
DATA_PATH = 'data.pickle'
ICD10_DESC_PATH = 'icd10cm_order_2021.txt'

DEFAULT_EMBEDDING_PARAMS = {
    "vector_size": 100,
    "window_size": 5,
    "skip_gram": 1,
    "min_count": 1
}

