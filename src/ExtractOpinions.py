# import StringDouble
# import ExtractGraph
import urllib.request, json
import stanfordnlp


class ExtractOpinions:
    # Extracted opinions and corresponding review id is saved in extracted_pairs, where KEY is the opinion and VALUE is the set of review_ids where the opinion is extracted from.
    # Opinion should in form of "attribute, assessment", such as "service, good".
    extracted_opinions = []

    def __init__(self):
        file = open("assign4_reviews.txt", "r")
        result = []
        input_text = file.readlines()
        # stanfordnlp.download('en')
        nlp = stanfordnlp.Pipeline()
        for i in range(0, len(input_text) - 1):
            doc = nlp(input_text[i])
            midresult = doc.conll_file.conll_as_string()
            midresult = midresult.split('\n')
            last_PROPN = ''
            last_Adj = ''
            for x in midresult:
                x = x.split('\t')
                if (len(x) > 4 and x[3] != 'PUNCT'):
                    if (x[3] == 'PROPN' or x[3] == 'NOUN'):
                        last_PROPN = last_PROPN + " " + x[1]
                        splitted = last_PROPN.split(" ")
                        if len(splitted) > 2:
                            splitted = splitted[1:]
                        last_PROPN = " ".join(splitted)
                    if (x[3] == 'ADJ'):
                        if (last_PROPN != ' '):
                            result.append((last_PROPN, x[1]))
                            last_PROPN = ''
            self.result = result
            return

    def extract_pairs(self, review_id, review_content):
        return self.result


#ExtractOpinions().extract_pairs(1, 2)