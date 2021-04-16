'''
Pipeline:
1. Break up Hashtags
2. Normalize
3. Contract
4. MMST spelling correction
'''

import os

from hashtag_split import HashtagSplit
from normalize import Normalize
from contract import Contract
from spelling_correction_mmst import SpellingCorrectionMMST


class PipelineMMST:

    '''
    Give an array of input and output paths.
    The two array must have same length, every input file will be
    processed by pipeline and fully processed file will be written to
    output_path at same array index.
    '''
    def process(self, input_paths, output_paths):
        # Init steps
        hs = HashtagSplit()
        nr = Normalize()
        ct = Contract()
        sm = SpellingCorrectionMMST()

        # execute pipeline
        for input_path, output_path in zip(input_paths, output_paths):
            # data paths
            path_0 = input_path
            path_1 = output_path[:-4] + '_1' + output_path[-4:]
            path_2 = output_path[:-4] + '_2' + output_path[-4:]
            path_3 = output_path[:-4] + '_3' + output_path[-4:]
            path_4 = output_path

            # set paths
            hs.set_paths(path_0, path_1)
            nr.set_paths(path_1, path_2)
            ct.set_paths(path_2, path_3)
            sm.set_paths(path_3, path_4)

            # run
            print("starting with " + os.path.basename(input_path))
            hs.run()
            print(os.path.basename(input_path) + ": hashtag done.")
            nr.run()
            print(os.path.basename(input_path) + ": normalize done.")
            ct.run()
            print(os.path.basename(input_path) + ": contract done.")
            sm.run()
            print(os.path.basename(input_path) + ": spell corr done.")


# driver code
if __name__ == '__main__':
    pmmst = PipelineMMST()
    file_path = os.path.dirname(os.path.abspath(__file__))

    inp_dir = os.path.join(file_path, '../data/cleaned')
    inp_neg = os.path.join(inp_dir, 'train_neg.txt')
    inp_pos = os.path.join(inp_dir, 'train_pos.txt')
    inp_test = os.path.join(inp_dir, 'test.txt')

    out_dir = os.path.join(file_path, '../data/pipeline_mmst')
    out_neg = os.path.join(out_dir, 'train_neg.txt')
    out_pos = os.path.join(out_dir, 'train_pos.txt')
    out_test = os.path.join(out_dir, 'test.txt')

    pmmst.process([inp_neg, inp_pos, inp_test], [out_neg, out_pos, out_test])
