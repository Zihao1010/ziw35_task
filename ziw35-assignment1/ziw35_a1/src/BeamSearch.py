import StringDouble
import heapq
import math

class BeamSearch:

    graph = []

    def __init__(self, input_graph):
        self.graph = input_graph
        return

    def beamSearchV1(self, pre_words, beamK, maxToken):
        # Basic beam search.
        pre_score = 0.0
        prev_beam = Beam(beamK)
        pre_words_list = pre_words.split()
        for i in range(len(pre_words_list)-1):
            pre_score += math.log(self.graph.getProb(pre_words_list[i], pre_words_list[i+1]))
        prev_beam.add(pre_score, False, pre_words_list)

        while True:
            cur_beam = Beam(beamK)

            for (prefix_score, complete, prefix) in prev_beam:
                if complete is True:
                    cur_beam.add(prefix_score, True, prefix)
                else:
                    next_word_dict = self.graph.graph[prefix[-1]]
                    for next_word, next_prob in next_word_dict.items():
                        if next_word == "</s>":
                            cur_beam.add(prefix_score+math.log(next_prob), True, prefix+[next_word])
                        else:
                            cur_beam.add(prefix_score+math.log(next_prob), False, prefix+[next_word])
            (best_score, best_complete, best_prefix) = max(cur_beam)
            if best_complete is True or len(best_prefix) == maxToken:
                sentence = " ".join(best_prefix)
                probability = best_score
                return StringDouble.StringDouble(sentence, probability)
            prev_beam = cur_beam

    def beamSearchV2(self, pre_words, beamK, param_lambda, maxToken):
        pre_score = 0.0
        prev_beam = Beam(beamK)
        pre_words_list = pre_words.split()
        for i in range(len(pre_words_list) - 1):
            pre_score += (1/math.pow(i+1, param_lambda))*math.log(self.graph.getProb(pre_words_list[i], pre_words_list[i + 1]))
        prev_beam.add(pre_score, False, pre_words_list)

        while True:
            cur_beam = Beam(beamK)

            for (prefix_score, complete, prefix) in prev_beam:
                if complete is True:
                    cur_beam.add(prefix_score, True, prefix)
                else:
                    next_word_dict = self.graph.graph[prefix[-1]]
                    for next_word, next_prob in next_word_dict.items():
                        if next_word == "</s>":
                            cur_beam.add(prefix_score + (1/math.pow(len(prefix)+1, param_lambda))*math.log(next_prob), True, prefix + [next_word])
                        else:
                            cur_beam.add(prefix_score + (1/math.pow(len(prefix + [next_word]), param_lambda))*math.log(next_prob), False, prefix + [next_word])
            (best_score, best_complete, best_prefix) = max(cur_beam)
            if best_complete is True or len(best_prefix) == maxToken:
                sentence = " ".join(best_prefix)
                probability = best_score
                return StringDouble.StringDouble(sentence, probability)
            prev_beam = cur_beam


class Beam:

    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width

    def add(self, prob, complete, prefix):
        heapq.heappush(self.heap, (prob, complete, prefix))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)