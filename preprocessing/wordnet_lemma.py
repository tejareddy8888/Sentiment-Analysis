from preprocessing_interface import PreprocessingInterface

import nltk
from nltk.stem import WordNetLemmatizer


class WordNetLemma(PreprocessingInterface):

    def run(self):
        super().run();

        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')

        lemmatizer = WordNetLemmatizer()

        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
                for line in input:
                    for word in line.split():
                        output.write(lemmatizer.lemmatize(word) + ' ')

                    output.write('\n')

        output.close()
