import argparse
from src.reader import Reader
from src.embedding_handler import EmbeddingHandler
from constants import DEFAULT_EMBEDDING_PARAMS
def main():
    parser = argparse.ArgumentParser(description="Interact with ICD Word Embeddings Project")

    parser.add_argument('-d', metavar='dir', help='Directory to which all data will be read or written', required=True)


    # read args
    parser.add_argument('--read', action='store_true',
                        help="Read data in from input file and save to destination file.")
    parser.add_argument('-query', help='Do you want to query Wikipeda & PubMed for extra text? (Default=True)',
                        required=False, default=1, type=int)
    # train args
    parser.add_argument('--train', action='store_true', help="Retrain the word2vec model.")
    parser.add_argument('-vector_size', help='Size of embeddings', required=False,
                        default=DEFAULT_EMBEDDING_PARAMS['vector_size'], type=int)
    parser.add_argument('-min_count', help='Minimum word count', required=False,
                        default=DEFAULT_EMBEDDING_PARAMS['min_count'], type=int)
    parser.add_argument('-window_size', help='Context window for a single word', required=False,
                        default=DEFAULT_EMBEDDING_PARAMS['window_size'], type=int)
    parser.add_argument('-sg', help='1 to use skipgram, 0 to use CBOW', required=False,
                        default=DEFAULT_EMBEDDING_PARAMS['skip_gram'], type=int)

    # predict args
    parser.add_argument('--predict', action='store_true', help="Load saved model and provide ICD10 codes for input.")
    parser.add_argument('-data_dir', help='Location of data to use. Default is data/',
                        required=False, default='/data', type=str)
    parser.add_argument('-code', metavar='ICD10 Code',
                        help='ICD10 code which you would like to predict on', required=False)
    parser.add_argument('-n', help='Number of most similar words to return', required=False, default=5, type=int)
    parser.add_argument('-weight_method', help='Function name for weighting description words',
                        required=False, default='simple', type=str)


    args = parser.parse_args()
    if not args.d:
        raise RuntimeError("You must specify a data directory. (-dir)")
    elif args.read:
        # read data
        reader = Reader(args.d, args.query)
        reader.read_icd()
        print("Finished reading ICD10 file")
        return 1
    elif args.train:
        handler = EmbeddingHandler()
        handler.train(args.d, min_count=args.min_count, size=args.vector_size, window=args.window_size, sg=args.sg)
        print(f"Model has been trained. Word vectors saved to {args.d}")
        return 1
    elif args.predict:
        if not args.code:
            raise RuntimeError("Please specify the ICD10 for which to retrieve similar words. (-code)")
        handler = EmbeddingHandler()
        most_similar_words = handler.predict(args.d, args.code, args.n, args.weight_method)
        print(f"Results for top {args.n} similar words using weighting_method={args.weight_method}:\n")
        max_len = max([len(w[0]) for w in most_similar_words])
        for i, pair in enumerate(most_similar_words):
            print(f"\t#{i+1}:\t{pair[0].ljust(max_len)}\t\tsimilarity={pair[1]}")


if __name__ == "__main__":
    main()

