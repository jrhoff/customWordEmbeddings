import re
import wikipediaapi
import pysbd
import pickle
import requests
from xml.etree import ElementTree
from math import log10
from tqdm import tqdm
from gensim.parsing.preprocessing import remove_stopwords
from constants import PUNCT, PARENT_CODE_LENGTH, CODE2DESC_PATH, WORD2TF_PATH, WORD2DF_PATH, FAMILY2TF_PATH, DATA_PATH, ICD10_DESC_PATH


def simple_clean(sentence):
    """
    Remove all punctuation, preserve '/' by replacing all punctuation with space so we don't lose words.
    Also lower case everything.
    """
    return re.sub(f"[{PUNCT}]", ' ', remove_stopwords(sentence).strip().lower()).replace('  ', ' ')


class Reader:
    def __init__(self, data_dir, query):
        self.data_dir = data_dir
        self.query = query
        self.data = []  # word2vec expects [[str]]
        self.code2desc = {}
        self.family2tf = {}  # holds term frequencies of words given a parent code
        self.word2df = {}
        self.word2tf = {}
        self.n_words = 0
        self.n_desc = 0

    def query_wikipedia(self, wiki, query):
        """
        Given a description as a query, see if there is a matching page result on wikipedia,
            if so, grab the summary for supplemental text
        """
        page = wiki.page(query)

        if page.exists():
            return [page.summary]
        return ''

    def query_pubmed(self, query, n_articles=3):
        """
        Given a description as a query, grab the abstracts of the 3 top search results from pubmed for supplemental data
        """
        try:
            URL = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&&retmax={n_articles}'
            resp = requests.get(URL)
            ids = [id.text for id in ElementTree.fromstring(resp.content)[3]]
            URL = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(ids)}&retmode=xml&rettype=abstract"
            resp = requests.get(URL)
            articles = [article[0][2][4][0].text for article in ElementTree.fromstring(resp.content)]
            return articles
        except:
            return []


    def process_family_frequencies(self, parent, family):
        """
        For a given parent, calculate save the term frequencies of the tokens in all child descriptions.
        Using descriptions as documents, also increment document frequency counts.
        Add all descriptions to self.data
        """
        if not family:
            return None
        counts = {}
        n_tokens = 0
        for description_array in family:
            seen = set([])
            for token in description_array:
                counts[token] = counts.get(token, 0) + 1
                n_tokens += 1

                self.word2tf[token] = self.word2tf.get(token, 0) + 1
                self.n_words += 1
                if token not in seen:
                    self.word2df[token] = self.word2df.get(token, 0) + 1
                    seen.add(token)

        self.family2tf[parent] = {tok: count/n_tokens for tok, count in counts.items()}
        self.data.extend(family)
        return True

    def process_articles(self, articles):
        """
        Given a list of article texts, pre-process them and add sentence vectors to self.data.
        """
        segmenter = pysbd.Segmenter(language="en", clean=False)
        processed = []
        for article in tqdm(articles, desc="Preprocessing Supplemental Articles"):
            processed.extend([simple_clean(sent).split() for sent in segmenter.segment(article)])
        return processed

    def dump(self):
        """
        Save pickle files for all attributes we may want to access quickly and/or individually later
        """
        # dump self.data
        pickle.dump(self.data, open(self.data_dir + DATA_PATH, 'wb+'))
        # dump self.code2desc
        pickle.dump(self.code2desc, open(self.data_dir + CODE2DESC_PATH, 'wb+'))
        # dump self.family2tf
        pickle.dump(self.family2tf, open(self.data_dir + FAMILY2TF_PATH, 'wb+'))
        # dump self.word2tf
        pickle.dump(self.word2tf, open(self.data_dir + WORD2TF_PATH, 'wb+'))
        # dump self.word2df
        pickle.dump(self.word2df, open(self.data_dir + WORD2DF_PATH, 'wb+'))
        return None

    def read_icd(self):
        """
        Read through the icd10 descriptions file and generate training data.
        - We'll consider a "family" to be the set of descriptions that share a 3 character prefix.
            Using 4 characters as a grouping prefix makes too many requests to fetch articles and takes much longer
        - When we find a 3 letter code, we'll query Wikipedia and PubMed using the code's description to try and get
             some extra data for our embeddings.
        - Keep track of all other codes that occur in each family. Once we hit the start of the next family, calculate
            metrics for the current family such as term frequency within the family and counts for document frequency.
        - Lastly, preprocess all the articles found, add them to our training data & pickle dump results
        """
        wiki = wikipediaapi.Wikipedia('en')  # may as well declare this here so I don't need to call it every query
        supplemental_articles = []
        with open(ICD10_DESC_PATH, 'r') as f:
            current_family = []     # list of lists of descriptions within the current family (3 letter code = family)
            current_parent = None   # Most recent 3 letter code seen
            for line in tqdm(f.readlines(), desc="ICD10 Lines Processed"):

                code = line[6:14].strip().lower()
                description = simple_clean(line[77:])
                self.code2desc[code] = description.split()

                if len(code) == PARENT_CODE_LENGTH:  # found a parent
                    # query web if set params to True
                    wiki_result = self.query_wikipedia(wiki, description) if self.query else []
                    pubmed_result = self.query_pubmed(description) if self.query else []

                    # store results
                    if wiki_result:
                        supplemental_articles.extend(wiki_result)
                    if pubmed_result:
                        supplemental_articles.extend(pubmed_result)

                    # update metrics using current family
                    self.process_family_frequencies(current_parent, current_family)
                    current_family = []
                    current_parent = code
                current_family.append(description.split())
                self.n_desc += 1

        # process the last family
        self.process_family_frequencies(current_parent, current_family)
        # go through all the articles we found, preprocess, and add to self.data
        self.data.extend(self.process_articles(supplemental_articles))

        # lastly calculate tf and idf over all descriptions (not including articles here) for use in weighting later
        self.n_words = log10(self.n_words)
        self.n_desc = log10(self.n_words)
        self.word2tf = {word: log10(count) - self.n_words for word, count in self.word2tf.items()}
        self.word2df = {word: count - self.n_desc for word, count in self.word2df.items()}
        self.dump()


