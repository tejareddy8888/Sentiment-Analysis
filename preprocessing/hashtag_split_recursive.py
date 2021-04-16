# Based on https://github.com/matchado/HashTagSplitter
'''
NOTE: This HashtagSplit has very poor performance.
 * It cannot find a split for most hashtags (including all that have numbers or typos)
 * It cannot distinguish between how good different split possibilities are.
   This leads to splits like: #awake -> #aw ake. Most splits are at wrong places.
 * It is extremely slow.
It is just included in case we need to build on it since it has very simple structure.
Use HashtagSplitWW instead.
'''

from preprocessing_interface import PreprocessingInterface
import nltk
from nltk.corpus import words, brown

class HashtagSplitRecursive(PreprocessingInterface):

    def split_hashtag_all_possibilities(self, hashtag):
        all_possibilities = []

        split_posibility = [hashtag[:i] in self.word_dictionary for i in reversed(range(len(hashtag)+1))]
        possible_split_positions = [i for i, x in enumerate(split_posibility) if x == True]

        for split_pos in possible_split_positions:
            split_words = []
            word_1, word_2 = hashtag[:len(hashtag)-split_pos], hashtag[len(hashtag)-split_pos:]

            if word_2 in self.word_dictionary:
                split_words.append(word_1)
                split_words.append(word_2)
                all_possibilities.append(split_words)

                another_round = self.split_hashtag_all_possibilities(word_2)

                if len(another_round) > 0:
                    all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]
            else:
                another_round = self.split_hashtag_all_possibilities(word_2)

                if len(another_round) > 0:
                    all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]

        return all_possibilities


    def run(self):
        super().run();

        # init dict
        try:
            nltk.data.find('words')
        except LookupError:
            nltk.download('words')

        self.word_dictionary = list(set(words.words()))

        for alphabet in "bcdefghjklmnopqrstuvwxyz":
            self.word_dictionary.remove(alphabet)


        # split hashtags
        no_split = 0
        n_pos = 0
        total = 0

        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
            for line in input:
                for word in line.split():
                    if word[0] == '#':

                        total += 1
                        all_pos = self.split_hashtag_all_possibilities(word[1:])

                        if len(all_pos) == 0:
                            #print("Couldn't split " + word)
                            no_split += 1
                        else:
                            if len(all_pos) > 1:
                                n_pos += 1
                                #print("Hashtag split: " + str(len(all_pos)) + " possibilities")

                            for split_word in all_pos[0]:
                                output.write(split_word + ' ')

                output.write('\n')

        output.close()

        print("Couldn't split " + no_split + "/" + total)
        print("Several possib  " + n_pos + "/" + total)
        print("not good: " + (no_split+n_pos) + "/" + total)
