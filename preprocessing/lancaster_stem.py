'''
LancasterStemmer is simple, but heavy stemming due to iterations
and over-stemming may occur. Over-stemming causes the stems to be not
linguistic, or they may have no meaning.
'''

from preprocessing_interface import PreprocessingInterface

from nltk.stem import LancasterStemmer


class LancasterStem(PreprocessingInterface):

    def run(self):
        super().run();

        # stem words in input file
        stemmer = LancasterStemmer()

        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
                for line in input:
                    for word in line.split():
                        output.write(stemmer.stem(word) + ' ')

                    output.write('\n')

        output.close()
