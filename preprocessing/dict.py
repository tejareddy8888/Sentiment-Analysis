from preprocessing_interface import PreprocessingInterface

import os
import urllib.request
import urllib3
from bs4 import BeautifulSoup
import csv
import json
import nltk.collocations
import collections
import nltk
from nltk.corpus import stopwords

class Dict:

    def __init__(self,
        emoticon_path='dicts/emoticon_dict.csv',
        slang_path='dicts/slang_dict.json',
        bigram_data='../data/snap_1.txt'):

        file_path = os.path.dirname(__file__)
        self.emoticon_path = os.path.join(file_path, emoticon_path)
        self.slang_path = os.path.join(file_path, slang_path)
        self.bigram_data = bigram_data
        self.bigram_path = file_path + '/dicts/bigram_snap.json'


    def scrape_emoticon(self):
        # get wikipedia tables
        url = 'https://en.wikipedia.org/wiki/List_of_emoticons'
        page = urllib.request.urlopen(url)

        soup = BeautifulSoup(page, "lxml")
        tables = soup.findAll('table', class_='wikitable')[:-3]
        tables.remove(tables[3])

        # extract emoticon and description
        emoticon = []
        description = []

        for table in tables:
            # remove all links
            for s in table.findAll('a'):
                s.replaceWithChildren()
            for s in table.findAll('sup', class_=True):
                s.extract()

            # add emoticons and description
            ignore = ['n/a', 'shocked', 'cup of tea']

            for row in table.findAll('tr'):
                cells = row.findAll('td')
                if len(cells) >= 3:
                    for i in range(len(cells)-2):
                        emoticon_string = cells[i].find(text=True).replace('\n', '').lower()
                        description_string = cells[-1].find(text=True).replace('\n', '').lower()

                        if description_string not in ignore:
                            single_emoticons = emoticon_string.split(' ')
                            for e in single_emoticons:
                                if len(e) != 0:
                                    emoticon.append(e)
                                    description.append(description_string)

        # clean
        for i in range(len(description)):
            # remove everything after ",", " or ", ". "
            description[i] = description[i].split(",", 1)[0]
            description[i] = description[i].split(" or ", 1)[0]
            description[i] = description[i].split(". ", 1)[0]

            # handle .. (some emoticons contain .. indicating symbol is repeated)
            if emoticon[i].endswith('..'):
                emoticon[i] = emoticon[i][:-2]
                emoticon.append(emoticon[i] + emoticon[i][-1]) # append with last sign once more
                description.append(description[i])
                emoticon.append(emoticon[-1] + emoticon[-1][-1]) # append with last sign twice more
                description.append(description[i])

            # add nose-less version of all emoticons: :-) -> :)
            no_nose = emoticon[i].split("‑")
            if len(no_nose) == 2:
                emoticon.append(no_nose[0] + no_nose[1])
                description.append(description[i])
            else:
                # Wikipedia used to different - for the noses
                no_nose = emoticon[i].split("-")
                if len(no_nose) == 2:
                    emoticon.append(no_nose[0] + no_nose[1])
                    description.append(description[i])

        # print
        for i in range(len(emoticon)):
            print(emoticon[i] + " " + description[i])

        # save
        w = csv.writer(open(self.emoticon_path, 'w'))
        for i in range(len(emoticon)):
            w.writerow([emoticon[i], description[i]])


    def scrape_slang(self):
        slang_dict = {}

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
                slang_dict[i.find('span').text[:-2]] = slang

        with open(self.slang_path, 'w+') as file:
            json.dump(slang_dict, file)


    def get_emoticon(self):
        # scrape emoticon dict if not already done
        if not os.path.isfile(self.emoticon_path):
            print('scraping emotions ...')
            self.scrape_emoticon()

        # get emoticon dict
        with open(self.emoticon_path, mode='r', encoding='utf8') as f:
            reader = csv.reader(f)
            dict = {rows[0]:rows[1] for rows in reader}

        return dict


    def get_slang(self):
        # init slang dict
        if not os.path.isfile(self.slang_path):
            print('scraping slang dict ...')
            self.scrape_slang()

        with open(self.slang_path,'r', encoding='utf8') as file:
            dict = json.loads(file.read())

        return dict

    def get_stopwords(self):
        stop_words = set(stopwords.words('english'))
        stop_words.add('<user>')
        stop_words.add('<url>')
        return stop_words


    def get_bigrams(self):
        if not os.path.isfile(self.bigram_path):
            print('getting bigram likelihoods ...')

            # get list of bigrams
            bgm = nltk.collocations.BigramAssocMeasures()
            file_content = open(self.bigram_data, 'r')
            tokens = [nltk.word_tokenize(line) for line in file_content]
            finder = nltk.collocations.BigramCollocationFinder.from_documents(tokens)
            bigram = finder.score_ngrams(bgm.likelihood_ratio)

            # turn into dict and dump to json
            dict = {(tup[0][0] + ' ' + tup[0][1]):tup[1] for tup in bigram}
            for i, tup in enumerate(bigram):
                print(tup)
                if i == 100:
                    break
            with open(self.bigram_path, 'w+') as file:
                json.dump(dict, file)

        else:
            with open(self.bigram_path,'r', encoding='utf8') as file:
                dict = json.loads(file.read())

        return dict
