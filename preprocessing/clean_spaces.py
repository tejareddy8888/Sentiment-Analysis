'''
standardizes spacing of special characters
e.g  "sentence?" -> "sentence ?"
     ": ' - )"   -> ":'-)'"
'''

from preprocessing_interface import PreprocessingInterface
from dict import Dict
import os


class CleanSpaces(PreprocessingInterface):

    def run(self):
        super().run();

        spec_sign = "[@_!#$%^&*()<>?/\|}{~:];'-"

        # init emoticon dict
        d = Dict()
        emot_dict = d.get_emoticon()

        # normalize words
        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
            for line in input:
                words = line.split()

                i = 0
                while i < len(words):
                    word = words[i]

                    # remove emoticon spacing
                    word_given = ''
                    word_nospace = ''
                    while word in spec_sign and len(word) == 1 and i < len(words):
                        word = words[i]
                        word_given += word + ' '
                        word_nospace += word
                        if word_nospace in emot_dict:
                            output.write(word_nospace + ' ')
                            word_given = ''
                            word_nospace = ''
                        i += 1

                    if len(word_given) > 0:
                        output.write(word_given)

                    else:
                        # question mark
                        if word[-1] == '?':
                            word = word[:-1] + ' ' + '?'

                        output.write(word + ' ')
                        i += 1

                output.write('\n')

        output.close()
