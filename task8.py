from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)  
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)   # { named_entity: subtree[1] }
                current_chunk = []
        else:
            continue
    return continuous_chunk

text = "Mark Norkin and John Spenser are working in New York."

print(get_continuous_chunks(text))