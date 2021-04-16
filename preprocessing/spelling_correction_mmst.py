import os
import sys

from preprocessing_interface import PreprocessingInterface
import enchant
from dict import Dict
import multiprocessing as mp
import threading
from math import floor

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, '../embed'))
from mmst import MMST
from embeddings import Loader


class SpellingCorrectionMMST(PreprocessingInterface):

    def __init__(self):
        self.nb = 0
        self.cores = mp.cpu_count()
        self.load = Loader()
        self.load.loadGloveModel()


    def prep_input(self):
        sentences = []
        with open(self.input, mode='r') as input:
            for line in input:
                sentences.append(line)
                self.nb += 1

        split_size = int(self.nb/self.cores)
        for i in range(self.cores-1):
            with open(self.input + '_' + str(i), "w+") as f:
                for j in range(i*split_size, (i+1)*split_size):
                    f.write(sentences[j])

        i = self.cores-1
        with open(self.input + '_' + str(i), "w+") as f:
            for j in range(i*split_size, self.nb):
                f.write(sentences[j])


    def merge_outputs(self):
        out = open(self.output, "w+")
        for i in range(self.cores):
            with open(self.output + '_' + str(i), "r") as f:
                for line in f:
                    out.write(line)

            #os.remove(self.input + '_' + str(i))


    def checker(self, id, d, slang_dict, stop_words, emoji_dict):
        first = True
        g = MMST(d, slang_dict, stop_words, emoji_dict)
        input = open(self.input + '_' + str(id), "r")
        '''
        output_r = open(self.output + '_' + str(id), "r+")
        already_written = 0
        for line in output_r:
            already_written += 1
        output_r.close()
        '''

        prog = 0
        with open(self.output + '_' + str(id), "a+") as f:
            for line in input:
                '''
                prog += 1
                if prog <= already_written:
                    continue
                '''
                if first:
                    print(line)
                    first = False
                try:
                    tmp = g.input_sentence(line, self.load, verbose=False)
                    f.write(tmp)
                except IndexError:
                    print("error " + line)

    def run(self):
        super().run()

        dict = Dict()
        slang_dict = dict.get_slang()
        stop_words = dict.get_stopwords()
        emoji_dict = dict.get_emoticon()
        d = enchant.Dict("en_US")

        self.prep_input()

        # dictionnary defined in MMST __init___
        share = floor(self.nb / self.cores)

        ts = [threading.Thread(target=self.checker, args=(i, d, slang_dict, stop_words, emoji_dict)) for i in range(self.cores)]

        for t in ts:
            t.start()

        for t in ts:
            t.join()

        print("merging")

        self.merge_outputs()
