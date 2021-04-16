## Preprocessing
#### Preprocessing steps
All preprocessing steps implement the (informal) interface ```preprocessing_interface.py```. Each preprocessing step thus implements these two methods:
 - ```set_paths(input, output)```: Set paths for input and output files of step (inherited)
 - ```run()```: Process file at input path and write to file at output path

**Classes used for data cleaning:**
 - ```RemoveDuplicate```: Remove duplicate Tweets.
 - ```CleanSpaces```: Enforce space between sentence and punctuation. E.g. "Is it sunny?"" &rightarrow; "Is it sunny ?". Make sure emoticons don't contain spaces inbetween characters. E.g. : ' - )  &rightarrow;  :'-)

Further:
 - ```TagRemove```: Remove <url> and <user> tags
 - ```RemoveId```: Remove the IDs preceeding test set samples. E.g. "1,first test sample" &rightarrow; "first test sample"

**Classes used for preprocessing:**
 - ```SpellingCorrectionMMST```: Multi-threaded execution of MMST spelling correction.
 - ```SpellingCorrectionEnchant```: pyenchant spelling corrector
 - ```Contract```: Contract words with appostrophe. E.g. can't &arrowright; can not
 - ```EmoticonReplace```: Replace emoticons with their meaning. E.g. :) &rightarrow; "happy"
 - ```SlangReplace```: Replace slang and abbreviations with their standard forms. E.g. L8 &rightarrow; late
 - ```HashtagSplit``` (ww): Split hashtags into words. E.g. #pronetocry &rightarrow; "prone to cry"
 - ```Normalize```: Normalize words. E.g. "loooveee" &rightarrow; "love"

The dependencies of all above steps can be installed with ```requirements_prod.txt```.

For completeness, we also include alternative implementations and further steps, that we have tried out.

**Deprecated classes:**
 - ```HashtagSplitRecursive```: Naïve, recursive implementation of Hashtag split. Very slow.
 - ```SpellingCorrectionTextBlob```: Spelling corrector of popular TextBlob library. Very slow.
 - ```SpellingCorrectionSpellChecker```: Spelling corrector of SpellChecker library. Very slow

**Text normalization:**
 - ```StopwordsRemove```: Remove stop words
 - ```WordnetLemma```: Wordnet lemmatizer
 - ```PorterStem```: Porter stemmer
- ```LancesterStem```: Lancester stemmer

The dependencies of all preprocessing steps can be installed with ```requirements_dev.txt```.


#### Preprocessing pipelines
A pipeline apply all preprocessing steps by calling ```process(input_paths, output_paths)```. ```input_paths``` is a list of the files the pipeline should process and ```output_paths``` is a list containing the output paths of each processed file respectively.

We used three pipelines:
 - ```PipelineClean```: Cleans the raw data.
    1. ```RemoveDuplicate```
    2. ```CleanSpaces```
 - ```PipellineEnhance```: Combines all performance-enhancing preprocessing steps (excluding spelling correction) on the cleaned data.
    1. ```HashtagSplit```
    2. ```Normalize```
    3. ```Contract```
 - ```PipelineMMST```: Applies ```PipelineEnhance``` and MMST correction to cleaned data.
    1. ```HashtagSplit```
    2. ```Normalize```
    3. ```Contract```
    4. ```SpellingCorrectionMMST```

We include the outputs of all three pipelines in the data folder.


#### Helper classes
Additionally, we have the following helper classes:
 - ```Dict```: Scrapes emoticon and slang dict if needed and not locally available. ```is_word(word, eng=True, slang=True, emoticon=True)``` checks if word is a word in the dictionaries set to ```True```. Defining emoticons or slang as correct words, for example allows us to do normalizations like :)))) &rightarrow; :) or Llll8 &rightarrow; L8 (Late).
 - ```MMST```: Performs MMST spelling correction given a sentence.
 - ```GloVETools```: Creates Glove embeddings given a list of words. Also contains functions like ```plotTSNE()``` and ```plotTweet()``` that perform TSNE on the embeddings and plot them.
