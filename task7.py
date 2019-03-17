import re
import nltk

IN = re.compile(r'.*\bin\b(?!\b.+ing)')

# nltk.download('ieer')

for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
    for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus = 'ieer', pattern = IN):
        print(nltk.sem.rtuple(rel))
