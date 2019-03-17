import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

sentence = "Steve Key, Mark Norkin and John Spenser are working in the Westminster Christian Academy"

from nltk.tag import PerceptronTagger
from nltk.data import find

PICKLE = "averaged_perceptron_tagger.pickle"
AP_MODEL_LOC = "file:" + str(find('taggers/averaged_perceptron_tagger/' + PICKLE))

tagger = PerceptronTagger(load = False)
tagger.load(AP_MODEL_LOC)
pos_tag = tagger.tag

print(ne_chunk(pos_tag(word_tokenize(sentence))))