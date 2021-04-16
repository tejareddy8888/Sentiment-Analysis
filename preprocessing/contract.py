from preprocessing_interface import PreprocessingInterface

class Contract(PreprocessingInterface):

    '''
    dict from https://www.quora.com/Which-NLP-Tools-can-I-use-to-convert-sentences-like-Hes-rich-to-He-is-rich
    with slight modifications
    '''

    def __init__(self):
        '''
        TODO: Split 's to is but difficult because e.g. Justin's
        If they do it wrong. e.g. i'tll.
        '''

        self.dict = {"aren't" : "are not",
                        "can't" : "can not",
                        "cannot" : "can not",
                        "couldn't" : "could not",
                        "didn't" : "did not",
                        "doesn't" : "does not",
                        "don't" : "do not",
                        "hadn't" : "had not",
                        "hasn't" : "has not",
                        "haven't" : "have not",
                        "he'd" : "he would",
                        "he'll" : "he will",
                        "he's" : "he is",
                        "i'd" : "I would",
                        "i'd" : "I had",
                        "i'll" : "I will",
                        "i'm" : "I am",
                        "isn't" : "is not",
                        "it's" : "it is",
                        "it'll":"it will",
                        "i've" : "I have",
                        "let's" : "let us",
                        "mightn't" : "might not",
                        "mustn't" : "must not",
                        "shan't" : "shall not",
                        "she'd" : "she would",
                        "she'll" : "she will",
                        "she's" : "she is",
                        "shouldn't" : "should not",
                        "that's" : "that is",
                        "there's" : "there is",
                        "they'd" : "they would",
                        "they'll" : "they will",
                        "they're" : "they are",
                        "they've" : "they have",
                        "we'd" : "we would",
                        "we're" : "we are",
                        "weren't" : "were not",
                        "we've" : "we have",
                        "what'll" : "what will",
                        "what're" : "what are",
                        "what's" : "what is",
                        "what've" : "what have",
                        "where's" : "where is",
                        "who'd" : "who would",
                        "who'll" : "who will",
                        "who're" : "who are",
                        "who's" : "who is",
                        "who've" : "who have",
                        "won't" : "will not",
                        "wouldn't" : "would not",
                        "you'd" : "you would",
                        "you'll" : "you will",
                        "you're" : "you are",
                        "you've" : "you have",
                        "'re": " are",
                        "wasn't": "was not",
                        "we'll":" will",
                        "didn't": "did not",
                        "y'all": "you all",
                        "ya'll": "you all",
                        "how's": "how is",
                        "how'd": "how would",
                        "u'll": "you will",
                        "u're": "you are",
                        "u'd": "you would",
                        "u've": "you have",
                        "ain't": "is not",
                        "might've": "might have",
                        "here's": "here is",
                        "must've": "must have",
                        "should've": "should have",
                        "y'know": "you know",
                        "where'd": "where would",
                        "it'd": "it would",
                        "could've": "could have",
                        "would've": "would have",
                        "ma'am": "madam",
                        "that'd": "that would",
                        "that'll": "that will",
                        "there'll": "there will",
                        "there'd": "there would",
                        "nothing's": "nothing is",
                        "everything's": "everything is",
                        "karma's": "karma is",
                        "c'mon": "come on",
                        "why'd": "why would",
                        "needn't": "need not"
                    }

        additional_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not","didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}

        self.dict.update(additional_dict)

        for appo in list(self.dict):
            self.dict[appo.replace("'", "")] = self.dict[appo]

        self.dict.pop('ill', None)
        self.dict.pop('its', None)


    def print_dict(self):
        print("dict size: " + str(len(self.dict)))
        print(self.dict)


    def get_performance(self):
        # replace appos in input file
        print("Performance ApposRemove")

        # get dict size
        print("  Dict size: " + str(len(self.dict)))

        # process files
        from collections import defaultdict

        missed = defaultdict(int)
        replaced = 0
        not_rec = 0

        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
            for line in input:
                for word in line.split():
                    if word in self.dict:
                        replaced += 1
                        output.write(self.dict[word] + ' ')
                    elif word.find('\'') != -1:
                        # appos word that's not in dict
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
        print("Appos not recognized:")
        print(missed)


    def run(self):
        super().run();

        # replace appos in input file
        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
                for line in input:
                    for word in line.split():
                        if word in self.dict:
                            output.write(self.dict[word] + ' ')
                        else:
                            output.write(word + ' ')
                        '''
                        elif word.find('\'') != -1:
                            # appos word that's not in dict
                            print(word)
                            output.write(word + ' ')
                        '''

                    output.write('\n')

        output.close()
