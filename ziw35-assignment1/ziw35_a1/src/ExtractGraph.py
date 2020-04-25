

class ExtractGraph:

    # # key is head word; value stores next word and corresponding probability.
    # self.graph = {}
    graph = {}
    sentences_add = "data/assign1_sentences.txt"

    def __init__(self):

        # Extract the directed weighted graph, and save to {head_word, {tail_word, probability}}
        file = open(self.sentences_add, 'r')
        self.graph = {}
        for line in file.readlines():
            #print('reading line...')
            words = line.rstrip('\n').split()
            for i in range(len(words) - 1):
                if words[i] not in self.graph:
                    #print('  adding new word to graph: ', words[i])
                    temp = {}
                    temp[words[i + 1]] = 1
                    self.graph[words[i]] = temp  # dictonary add new element
                else:
                    word_dict = self.graph[words[i]]
                    if words[i + 1] not in word_dict:
                        word_dict[words[i + 1]] = 1
                    else:
                        word_dict[words[i + 1]] += 1
        #print('about to calculate prob...')
        for word in self.graph:
            s = sum(self.graph[word].values())  # sum of all word counts for 'word'
            for w in self.graph[word]:
                self.graph[word][w] = self.graph[word][w] / s
        return

    def getProb(self, head_word, tail_word):
        if head_word not in self.graph or tail_word not in self.graph[head_word]:
            return 0
        else:
            return self.graph[head_word][tail_word]




