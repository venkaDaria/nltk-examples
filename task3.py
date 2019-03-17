import nltk

text = '''London is considered to be one of the world's most important global cities 
and has been termed the world's most powerful, most desirable, most influential, most visited, most expensive, 
innovative, sustainable, most investment friendly, most popular for work, and the most vegetarian friendly city in the world. 
London exerts a considerable impact upon the arts, commerce, education, entertainment, fashion, finance, healthcare, media, 
professional services, research and development, tourism and transportation. London ranks 26 out of 300 major cities for economic  
performance.'''

# nltk.download('maxent_ne_chunker')
# nltk.download('words')

sentences = nltk.tokenize.sent_tokenize(text)
sentences = [nltk.tokenize.word_tokenize(sentence) for sentence in sentences]
tagged_sentence = nltk.pos_tag(sentences[0])
print(tagged_sentence)

tree = nltk.ne_chunk(tagged_sentence)
tree.draw()