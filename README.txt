Sample command for quick running with supplied data:

./predict.sh supplemental_data/ M24011 5 tfidf        # Returns top 5 most similar words to M240 using the tfidf weight scheme




Descriptions of different files...
# Read data with read.sh, it takes to arguments:
    1 - directory where processed data will be stored
    2 - Flag determining whether or not Wikipedia and PubMed will be queried while reading in codes | Acceptable values are {0,1}

    Example
            ./read.sh data/ 1          # Will query Wikipedia and PubMed for text for all 3 letter codes. Reading takes ~25 minutes
            ./read.sh data/ 0          # No querying, only use descriptions as data. Reads within seconds.

# Train word embeddings with train.sh, it takes 1 argument:
    1 - directory holding data

    Example:
        ./train.sh data/            # Saves word embeddings to data/

# Get top N similar words given an ICD code with predict.sh, it takes 4 arguments:
    1 - directory holding data
    2 - ICD10 code
    3 - N
    4 - Weighting method = {simple, family, tfidf}

    Example:
        ./predict.sh data/ M24011 5 simple          # Uses the data and word vectors in data/ to compute the top 5 similar words to M24011
                                                    # using simple weighting
PACKAGES USED:
--------------
pip install wikipedia
pip install pysbd
pip install tqdm
pip install gensim
pip install numpy