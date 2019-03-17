import nltk
from nltk.corpus import stopwords, reuters

print(stopwords.words("english"))

def content_fraction(text):
    stopwords_en = stopwords.words("english")
    content = [w for w in text if w.lower() not in stopwords_en]
    return len(content) / len(text)

print("***")

# nltk.download("reuters")
print(content_fraction(reuters.words()))