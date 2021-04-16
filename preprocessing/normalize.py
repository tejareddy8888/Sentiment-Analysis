'''
normalizes words
e.g  'llooooooovvvee' -> 'love'
'''

from preprocessing_interface import PreprocessingInterface
import enchant
from itertools import groupby
from dict import Dict
import os


class Normalize(PreprocessingInterface):

    def is_word(self, string):
        return self.en_dict.check(string) or string in self.slang_dict or string in self.emoticon_dict


    def get_norm_string(self, substrings, ind):
        cur_string = ''.join(substrings)

        if self.is_word(cur_string):
            return cur_string, True

        elif ind == len(substrings):
            return cur_string, False

        elif len(substrings[ind]) > 1:
            # try replace with one letter
            substrings[ind] = substrings[ind][0]
            candidate, is_word = self.get_norm_string(substrings, ind+1)
            if is_word:
                return candidate, True

            # try replace with 2 letters
            substrings[ind] += substrings[ind][0]
            candidate, is_word = self.get_norm_string(substrings, ind+1)
            if is_word:
                return candidate, True

            # return unaltered string
            return cur_string, False

        else:
            return self.get_norm_string(substrings, ind+1)


    def run(self):
        super().run();

        # init english dict
        self.en_dict = enchant.Dict("en_US")
        d = Dict()
        self.slang_dict = d.get_slang()
        self.emoticon_dict = d.get_emoticon()

        # normalize words
        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
            for line in input:
                for word in line.split():
                    if not self.en_dict.check(word):
                        l = [''.join(g) for _, g in groupby(word)]
                        if len(l) <= 10:
                            word, _ = self.get_norm_string(l, 0)

                    output.write(word + ' ')

                output.write('\n')

        output.close()
