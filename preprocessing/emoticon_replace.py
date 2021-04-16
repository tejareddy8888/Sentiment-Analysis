from preprocessing_interface import PreprocessingInterface
from dict import Dict


import os


class EmoticonReplace(PreprocessingInterface):

    def print_dict(self):
        if not os.path.isfile(self.dict_path):
            print('scraping ...')
            self.get_replace_words()
        else:
            with open(self.dict_path, mode='r') as f:
                reader = csv.reader(f)
                for rows in reader:
                    print(rows[0] + ", " + rows[1])

    def get_performance(self):
        print("Performance EmoticonReplace")

        # get dict size
        if not os.path.isfile(self.dict_path):
            print('scraping ...')
            self.get_replace_words()

        with open(self.dict_path, mode='r', encoding='utf8') as f:
            reader = csv.reader(f)
            for rows in reader:
                dict = {rows[0]:rows[1] for rows in reader}

        print("  Dict size: " + str(len(dict)))

        # process files
        import re
        from collections import defaultdict

        hit = defaultdict(int)
        missed = defaultdict(int)
        replaced = 0
        not_rec = 0

        regex = re.compile("[@_!#$%^&*()<>?/\|}{~:];'")
        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
                for line in input:
                    for word in line.split():
                        if word in dict:
                            hit[word] += 1
                            replaced += 1
                            output.write(dict[word] + ' ')
                        elif not regex.search(word) is None and len(word) > 1 and not word[0] == '#' and not (word[0] == '<' and word[-1] == '>'):
                            missed[word] += 1
                            not_rec += 1
                            output.write(word + ' ')
                        else:
                            output.write(word + ' ')

                    output.write('\n')

        output.close()

        print("  replaced: " + str(replaced))
        print("  not recognized: " + str(not_rec) + " (distinct: " + str(len(missed)) + ")")
        print()
        print("Emoticons not recognized:")
        print(missed)
        print()
        print("Emoticons replaced:")
        print(hit)


    def run(self):
        super().run();

        # get emoticon dict
        d = Dict()
        dict = d.get_emoticon()

        # replace emoticons in input file
        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
                for line in input:
                    for word in line.split():
                        if word in dict:
                            output.write(dict[word] + ' ')
                        else:
                            output.write(word + ' ')

                    output.write('\n')

        output.close()
