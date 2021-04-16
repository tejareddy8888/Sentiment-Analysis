from preprocessing_interface import PreprocessingInterface

class RemoveDuplicate(PreprocessingInterface):

    def run(self):
        super().run();

        # remove
        output = open(self.output, 'w+', encoding="utf8")
        with open(self.input, mode='r') as input:
            prev = next(input)
            output.write(prev)
            for line in input:
                if not line == prev:
                    prev = line
                    output.write(line)

        output.close()
