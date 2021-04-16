# Based on https://medium.com/@indreshbhattacharyya/remaking-of-shortened-sms-tweet-post-slangs-and-word-contraction-into-sentences-nlp-7bd1bbc6fcff
'''
Scrape slang words from https://www.noslang.com to handle words like 'omg',
'lol', 'dunno', etc.
Only replace words if they cannot be found in normal dict
'''

from preprocessing_interface import PreprocessingInterface
import enchant
import os
from dict import *

class SlangReplace(PreprocessingInterface):

    def run(self):
        super().run()

        # get dicts
        eng_dict = enchant.Dict("en_US")
        d = Dict()
        slang_dict = d.get_slang()

        # replace slang words
        corr_word_rep = ['dunno', 'gonna', 'rt']

        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
            for line in input:
                for word in line.split():
                    if word in corr_word_rep or (not eng_dict.check(word) and not word[0] == '#'):
                        if word in slang_dict:
                            word = slang_dict[word]


                    output.write(word + ' ')

                output.write("\n")

        output.close()
