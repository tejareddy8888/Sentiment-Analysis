from preprocessing_interface import PreprocessingInterface
from spellchecker import SpellChecker
import enchant
import json

class SpellingCorrectionSpellChecker(PreprocessingInterface):

    def is_word(self, string):
        return self.en_dict.check(string) or string in self.slang_dict


    def run(self):
        super().run();

        # init english dict
        self.en_dict = enchant.Dict("en_US")

        with open("dicts/slang_dict.json",'r', encoding='utf8') as file:
            self.slang_dict = json.loads(file.read())


        spell = SpellChecker()

        # correct
        output = open(self.output, 'w+')
        i = 0
        with open(self.input, mode='r') as input:
                for line in input:
                    for word in line.split():

                        if not (word[0] == '<' and word[-1] == '>') and not word[0] == "#" and not self.is_word(word):
                            word = spell.correction(word)

                        output.write(word + ' ')

                    output.write('\n')
                    i += 1
                    if i % 10000 == 0:
                        print(i)
        output.close()
