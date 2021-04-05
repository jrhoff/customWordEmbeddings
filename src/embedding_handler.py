import pickle
import numpy as np
import gensim
from gensim.models import Word2Vec
from tqdm import tqdm
from constants import WORD2DF_PATH, WORD2TF_PATH, FAMILY2TF_PATH, CODE2DESC_PATH, WORD_EMBEDDING_PATH, DATA_PATH


class EmbeddingHandler:
    def __init__(self):
        self.funcs = {'simple': self._simple_average,
                      'family': self._weight_by_family_frequency,
                      'tfidf': self._weight_by_tfidf
                      }

    def _simple_average(self, data_dir, component_embeddings, code, description):
        """
        Simply take the average of all description word embeddings along the 0 axis without any weighting by word
        """
        return np.mean(component_embeddings, 0)

    def _weight_by_family_frequency(self, data_dir, component_embeddings, code, description):
        """
        Weight the component embeddings using the term frequencies of the words in the description.
        The term frequencies we use here are calculated over the code family (first 3 chars of the code).
        """
        family2tf = pickle.load(open(data_dir + FAMILY2TF_PATH, 'rb'))
        family = code[:3].lower()
        tfs = np.array([[family2tf[family][word]] for word in description])
        return np.sum((component_embeddings*tfs), axis=0)

    def _weight_by_tfidf(self, data_dir, component_embeddings, code, description):
        """
        Weight the component embeddings using something like tf-idf. We don't really have documents here,
            but for this scheme we consider the tf over the given code's family and the df over all descriptions.
        tf -> frequency of word in this family
        df -> how many descriptions did this word appear in out of ALL the descriptions
        """
        family2tf = pickle.load(open(data_dir + FAMILY2TF_PATH, 'rb'))
        family = code[:3].lower()
        word2df = pickle.load(open(data_dir + WORD2DF_PATH, 'rb'))
        tfidfs = np.array([[family2tf[family][word]/word2df[word]] for word in description])
        return np.sum((component_embeddings*tfidfs), axis=0)

    def compute_icd10_embeddings(self, data_dir, code, word_embeddings, weighting_method):
        """
        Get the word embeddings for each word in the description and combine them using
            whichever weighting method was specified by the user.
        """
        try:
            weighting_method = self.funcs[weighting_method]
        except KeyError:
            raise KeyError(f"Specified function name {weighting_method} not found, please double-check")
        code2desc = pickle.load(open(data_dir + CODE2DESC_PATH, 'rb'))
        components = np.array([word_embeddings[description_word] for description_word in code2desc[code]])
        return weighting_method(data_dir, components, code, code2desc[code])

    def train(self, data_dir, min_count=1, size=100, window=5, sg=1):
        """
        Train word embeddings using Word2Vec. Only save the embedding dictionary.
        """
        data = pickle.load(open(data_dir + DATA_PATH, 'rb+'))
        model = Word2Vec(sentences=data, min_count=min_count, vector_size=size, window=window, sg=sg)
        model.wv.save(data_dir + WORD_EMBEDDING_PATH)
        print(f"Word embeddings saved to {data_dir + WORD_EMBEDDING_PATH}")
        return True

    def predict(self, path_to_data, code, words_to_report, weighting_method):
        """
        1 - Given an ICD10 code, calculate its embedding using the specified weighting method on its description.
        2 - Return the top # words_to_report most simliar words to the ICD10 embedding.
        """
        word_embeddings = gensim.models.KeyedVectors.load(path_to_data + WORD_EMBEDDING_PATH)
        icd10_embedding = self.compute_icd10_embeddings(path_to_data, code.lower(), word_embeddings,
                                                        weighting_method=weighting_method)
        sims = gensim.models.KeyedVectors.cosine_similarities(icd10_embedding, word_embeddings.vectors)
        by_word = sorted([(word_embeddings.index_to_key[i], sims[i])for i in range(len(sims))], key=lambda x: x[1])
        topn = by_word[::-1][:words_to_report]
        return topn
