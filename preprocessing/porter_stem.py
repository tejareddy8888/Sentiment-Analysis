from preprocessing_interface import PreprocessingInterface

from nltk.stem.porter import *


class PorterStem(PreprocessingInterface):

    def run(self):
        super().run();

        # stem words in input file
        stemmer = PorterStemmer()

        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
                for line in input:
                    for word in line.split():
                        output.write(stemmer.stem(word) + ' ')

                    output.write('\n')

        output.close()
