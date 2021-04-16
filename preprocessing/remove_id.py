'''
NOTE: This removes the Ids for the test_data. The data has the following format:
1,sea doo pro sea scooter ( sports with the portable sea-doo seascootersave air , stay longer in the water and ... <url>
Output of this class (remove '1,'):
sea doo pro sea scooter ( sports with the portable sea-doo seascootersave air , stay longer in the water and ... <url>
'''

from preprocessing_interface import PreprocessingInterface


class RemoveId(PreprocessingInterface):

    def run(self):
        super().run();

        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
            for line in input:
                line_new = line[(line.index(",")+1):]
                output.write(line_new)

        output.close()
