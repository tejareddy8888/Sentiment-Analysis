# informal interface for preprocessing steps

class PreprocessingInterface:

    # methods we want each step to have
    def set_paths(self, input, output):
        self.input = input
        self.output = output

    def get_performance(self):
        """print performance of step"""

        pass

    def run(self):
        """run preprocessing step"""

        # Code you want to execute at each step before run
        if self.input is None:
            raise Exception("Source not set", "You haven't set the input source.")
        elif self.output is None:
            raise Exception("Destination not set", "You haven't set the output destination.")

        pass
