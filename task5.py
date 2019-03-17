from nltk.stem.snowball import EnglishStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize

english_stemmer = EnglishStemmer()

class StemmedCountVectorizer(TfidfVectorizer):
    def build_anyalyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

stem_vectorizer = StemmedCountVectorizer(stop_words='english')   

text = '''London is considered to be one of the world's most important global cities 
and has been termed the world's most powerful, most desirable, most influential, most visited, most expensive, 
innovative, sustainable, most investment friendly, most popular for work, and the most vegetarian friendly city in the world. 
London exerts a considerable impact upon the arts, commerce, education, entertainment, fashion, finance, healthcare, media, 
professional services, research and development, tourism and transportation. London ranks 26 out of 300 major cities for economic  
performance.'''

sentences = sent_tokenize(text)

print(stem_vectorizer.get_stop_words())
print("***")
print(stem_vectorizer.fit_transform(sentences))
