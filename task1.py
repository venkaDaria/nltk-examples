import nltk

text = '''London is considered to be one of the world's most important global cities 
and has been termed the world's most powerful, most desirable, most influential, most visited, most expensive, 
innovative, sustainable, most investment friendly, most popular for work, and the most vegetarian friendly city in the world. 
London exerts a considerable impact upon the arts, commerce, education, entertainment, fashion, finance, healthcare, media, 
professional services, research and development, tourism and transportation. London ranks 26 out of 300 major cities for economic  
performance.'''

# разбиение текста на предложения
sentences = nltk.tokenize.sent_tokenize(text)

print(sentences)
print(sentences[0])

print("***")

# разбиение текста на слова
sentences = [nltk.tokenize.word_tokenize(sentence) for sentence in sentences]
print(sentences)

print("***")

# stemmer Porter (Snowball)
stemmer = nltk.stem.SnowballStemmer(language="english")
print([stemmer.stem(word) for word in sentences[0]])

print("***")

# lemmatizer WordNet 
# nltk.download("wordnet") - thesaurus, ontology
wnl = nltk.WordNetLemmatizer()
print([wnl.lemmatize(t) for t in sentences[0]])

print("***")

# stemmer Lancaster
tokens = text.split(' ')
lancaster = nltk.stem.LancasterStemmer() # nltk.LancasterStemmer()
print([lancaster.stem(t) for t in tokens])

print("***")