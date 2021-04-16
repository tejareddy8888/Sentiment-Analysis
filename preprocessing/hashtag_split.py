'''
Hashtag split based on word weighing algorithm.
'''

from preprocessing_interface import PreprocessingInterface
from wordsegment import load, segment

class HashtagSplit(PreprocessingInterface):

    def run(self):
        super().run();

        load()

        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
            for line in input:
                for word in line.split():
                    if word[0] == '#':
                        split = segment(word[1:])

                        for split_word in split:
                            output.write(split_word + ' ')

                    else:
                        output.write(word + ' ')

                output.write('\n')

        output.close()
