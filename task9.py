import nltk

sentence = "Mark Norkin and John Spenser are working in the Westminster Christian Academy"

for sent in nltk.sent_tokenize(sentence):
    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
        if hasattr(chunk, 'label'):
            print(chunk.label(), ''.join(c[0] for c in chunk))