'''
Pipeline:
1. Remove duplicate
2. Remove tags
3. Space clean
4. If test: removeId
'''
import os

from clean_spaces import CleanSpaces
from remove_duplicate import RemoveDuplicate
from tag_remove import TagRemove
from remove_id import RemoveId

class PipelineClean:

    '''
    Give an array of input and output paths.
    The two array must have same length, every input file will be
    processed by pipeline and fully processed file will be written to
    output_path at same array index.
    '''
    def process(self, input_paths, output_paths):
        # Init steps
        rd = RemoveDuplicate()
        tr = TagRemove()
        cs = CleanSpaces()
        ri = RemoveId()

        # execute pipeline
        for input_path, output_path in zip(input_paths, output_paths):
            # data paths
            path_0 = input_path
            path_1 = output_path[:-4] + '_1' + output_path[-4:]
            path_2 = output_path[:-4] + '_2' + output_path[-4:]
            path_3 = output_path[:-4] + '_3' + output_path[-4:]
            path_4 = output_path

            # set paths
            rd.set_paths(path_0, path_1)
            tr.set_paths(path_1, path_2)
            if input_path[-8:] == 'test.txt':
                cs.set_paths(path_2, path_3)
                ri.set_paths(path_3, path_4)
            else:
                cs.set_paths(path_2, path_4)

            # run
            print("starting with " + os.path.basename(input_path))
            rd.run()
            print(os.path.basename(input_path) + ": remove duplicates done.")
            tr.run()
            print(os.path.basename(input_path) + ": remove tags done.")
            cs.run()
            print(os.path.basename(input_path) + ": clean spaces done.")
            if input_path[-8:] == 'test.txt':
                ri.run()
                print(os.path.basename(input_path) + ": remove id done.")


# driver code
if __name__ == '__main__':
    pclean = PipelineClean()
    file_path = os.path.dirname(os.path.abspath(__file__))

    inp_dir = os.path.join(file_path, '../data/cleaned')
    inp_neg = os.path.join(inp_dir, 'train_neg_full.txt')
    inp_pos = os.path.join(inp_dir, 'train_pos_full.txt')
    inp_test = os.path.join(inp_dir, 'test_data.txt')

    out_dir = os.path.join(file_path, '../data/cleaned')
    out_neg = os.path.join(out_dir, 'train_neg.txt')
    out_pos = os.path.join(out_dir, 'train_pos.txt')
    out_test = os.path.join(out_dir, 'test.txt')

    pclean.process([inp_neg, inp_pos, inp_test], [out_neg, out_pos, out_test])
