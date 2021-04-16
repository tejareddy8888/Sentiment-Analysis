from preprocessing_interface import PreprocessingInterface
import nltk
from nltk.corpus import stopwords


class TagRemove(PreprocessingInterface):

    def run(self):
        super().run();

        tags = ['<url>', '<user>']

        # remove
        output = open(self.output, 'w+', encoding="utf8")
        with open(self.input, mode='r') as input:
                for line in input:
                    for word in line.split():
                        if not word in tags:
                            output.write(word + ' ')

                    output.write('\n')

        output.close()
