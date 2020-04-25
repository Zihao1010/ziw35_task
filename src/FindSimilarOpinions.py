import gensim.models.keyedvectors as word2vec
import ExtractOpinions

class FindSimilarOpinions:

    extracted_opinions = {}
    word2VecObject = []
    cosine_sim=0.5

    def __init__(self, input_cosine_sim, input_extracted_ops):
        self.cosine_sim = input_cosine_sim
        self.extracted_opinions = input_extracted_ops
        word2vec_add = 'data\\assign4_word2vec_for_python.bin'
        self.word2VecObject = word2vec.KeyedVectors.load_word2vec_format(word2vec_add, binary=True)
        print(self.word2VecObject.similarity('great', 'good'))

        # above line can be deleted in your real code, only for reference.
        return

    def findSimilarOpinions(self, query_opinion):
        # example data, which you will need to remove in your real code. Only for demo.
        similar_opinions  = []
        result = ExtractOpinions.ExtractOpinions().extract_pairs(1, 2)
        for tup in result:
            try:
                if self.word2VecObject.similarity(tup[1].lower().strip(), query_opinion.split(",")[1].strip()) > 0.5 : #self.
                    similar_opinions.append(', '.join(tup))
            except:
                pass
        return similar_opinions