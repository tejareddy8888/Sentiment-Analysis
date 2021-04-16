from preprocessing_interface import PreprocessingInterface
import os
import json
import enchant
from enchant.checker import SpellChecker

class SpellingCorrectionEnchant(PreprocessingInterface):

    '''
    file_len and print_prog serve to print progress. file_len is the amount of
    lines the file has, print prog is after how many percent points of progress,
    the progress should be printed.
    '''
    def __init__(self, print_prog=5, dict_path='slang_dict.json'):
        self.file_len = 1000000
        self.print_prog=5

        self.slang_dict = {}
        file_path = os.path.dirname(__file__)
        self.dict_path = os.path.join(file_path, dict_path)


    def get_dict(self):
        # Generate links for all slang a-z
        linkDict=[]
        for one in range(97,123):
            linkDict.append(chr(one))

        # scrape sites
        http = urllib3.PoolManager()

        for alpha in linkDict:
            r = http.request('GET','https://www.noslang.com/dictionary/' + alpha)
            soup = BeautifulSoup(r.data,'html.parser')

            for i in soup.findAll('div',{'class':'dictionary-word'}):
                slang = i.find('abbr')['title']
                self.slang_dict[i.find('span').text[:-2]] = slang

        with open(self.dict_path, 'w') as file:
            json.dump(self.slang_dict, file)


    def is_word(self, string):
        return self.en_dict.check(string) or string in self.slang_dict


    def run_batch(self):
        super().run()

        # stuff for printing progress
        prog = 0
        prog_perc = 0
        print_steps = self.file_len/100 * self.print_prog
        next_prog = print_steps

        # init english dict
        self.en_dict = enchant.Dict("en_US")

        # init slang dict
        if not os.path.isfile(self.dict_path):
            print('scraping ...')
            self.get_dict()

        with open(self.dict_path,'r', encoding='utf8') as file:
            self.slang_dict = json.loads(file.read())


        # process file
        spell = SpellChecker("en_UK")

        # count lines
        line_count = 0
        if os.path.isfile(self.output):
            with open(self.output, mode='r') as output:
                for line in output:
                    line_count += 1

            output.close()

        output = open(self.output, 'a+')
        with open(self.input, mode='r') as input:
            for i in range(line_count):
                next(input)
                prog += 1

            if prog > next_prog:
                prog_perc += self.print_prog
                next_prog += print_steps


            print('Starting at line ' + str(prog) + '. Progress: ' + str(prog_perc) + '% of ' + os.path.basename(self.input))
            prog_perc += self.print_prog

            for line in input:
                for word in line.split():
                    if not self.is_word(word):
                        spell.set_text(word)
                        for err in spell:
                            if len(err.suggest()) > 0:
                                word = err.suggest()[0]

                    output.write(word + ' ')
                output.write('\n')
                prog += 1

                if prog > next_prog:
                    print('Progress: ' + str(prog_perc) + '% of ' + os.path.basename(self.input))
                    prog_perc += self.print_prog
                    next_prog += print_steps

        output.close()
        print('Finished ' + self.input)


    def run_batch(self):
        super().run()

        # stuff for printing progress
        prog = 0
        prog_perc = 0
        print_steps = self.file_len/100 * self.print_prog
        next_prog = print_steps

        # init english dict
        self.en_dict = enchant.Dict("en_US")

        # init slang dict
        if not os.path.isfile(self.dict_path):
            print('scraping ...')
            self.get_dict()

        with open(self.dict_path,'r', encoding='utf8') as file:
            self.slang_dict = json.loads(file.read())


        # process file
        spell = SpellChecker("en_UK")

        # count lines
        line_count = 0
        if os.path.isfile(self.output):
            with open(self.output, mode='r') as output:
                for line in output:
                    line_count += 1

            output.close()

        output = open(self.output, 'a+')
        with open(self.input, mode='r') as input:
            for i in range(line_count):
                next(input)
                prog += 1

            if prog > next_prog:
                prog_perc += self.print_prog
                next_prog += print_steps


            print('Starting at line ' + str(prog) + '. Progress: ' + str(prog_perc) + '% of ' + os.path.basename(self.input))
            prog_perc += self.print_prog

            for line in input:
                for word in line.split():
                    if not self.is_word(word):
                        spell.set_text(word)
                        for err in spell:
                            if len(err.suggest()) > 0:
                                word = err.suggest()[0]

                    output.write(word + ' ')
                output.write('\n')
                prog += 1

                if prog > next_prog:
                    print('Progress: ' + str(prog_perc) + '% of ' + os.path.basename(self.input))
                    prog_perc += self.print_prog
                    next_prog += print_steps

        output.close()
        print('Finished ' + self.input)


    def run(self):
        super().run();

        spell = SpellChecker("en_UK","en_US")

        # correct
        output = open(self.output, 'w+')
        with open(self.input, mode='r', encoding='utf8') as input:
            for line in input:
                spell.set_text(line)
                for err in spell:
                    if len(err.suggest()) > 0:
                        sug = err.suggest()[-1]
                        err.replace(sug)

                line = spell.get_text()
                output.write(line)

        output.close()
