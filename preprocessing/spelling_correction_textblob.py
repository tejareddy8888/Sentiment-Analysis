from preprocessing_interface import PreprocessingInterface
import os
import json
import enchant
from textblob import TextBlob
import re

class SpellingCorrectionTextBlob(PreprocessingInterface):

    '''
    file_len and print_prog serve to print progress. file_len is the amount of
    lines the file has, print prog is after how many percent points of progress,
    the progress should be printed.
    '''
    def __init__(self, file_len=1000000, print_prog=5, dict_path='dicts/slang_dict.json', conf=0.51):
        self.file_len = 1000000
        self.print_prog=5

        self.slang_dict = {}
        file_path = os.path.dirname(__file__)
        self.dict_path = os.path.join(file_path, dict_path)
        self.conf = conf

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


        # count lines
        line_count = 0
        if os.path.isfile(self.output):
            with open(self.output, mode='r') as output:
                for line in output:
                    line_count += 1

            output.close()


        # process file
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

            regex = '[@_!#$%^&*()<>?/\|}{~:].,=+;"<\'-'

            for line in input:
                for word in line.split():
                    if not self.is_word(word) and word.isalpha():
                        print(word, end=' ')
                        blob = TextBlob(word)
                        sug = blob.words[0].spellcheck()
                        print(sug[0])
                        if len(sug) > 0:
                            conf = sug[0][1]
                            if conf > self.conf:
                                word = sug[0][0]

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

        # TODO

        self.en_dict = enchant.Dict("en_US")
        with open(self.dict_path,'r', encoding='utf8') as file:
            self.slang_dict = json.loads(file.read())

        # correct
        output = open(self.output, 'w+')
        i = 0
        with open(self.input, mode='r', encoding='utf8') as input:
            for line in input:
                for word in line.split():
                    if not self.is_word(word) and len(word) > 1 and word[0] != "#" and not word in ["i'm", "im"]:
                        blob = TextBlob(word)
                        if len(blob.words) > 0:
                            sugs = blob.words[0].spellcheck()
                            if len(sugs) > 0:
                                word = blob.words[0].spellcheck()[0][0]

                        '''
                        conf = sug[1]
                        #print(str(conf) + ' ' + sug[0])
                        print(sug)
                        print()
                        if conf > self.conf:
                            word = sug[0]
                        '''

                    output.write(word + ' ')
                output.write('\n')
                if i % 100000 == 0:
                    print("  {}".format(i))
                i += 1

        output.close()
