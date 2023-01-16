import spacy
from spacy import displacy
import en_core_web_sm
nlp = en_core_web_sm.load()

print(nlp.get_pipe("ner").labels)

text = 'a group of women sitting next to each other at a table, college girls, golondrinas, early 20s, college party, barney and friends, 2000s photo, cgs society, blue faces, mid-20s, photo 2'
doc = nlp(text)
print(doc)
for word in doc.ents:
    print(word.text,word.label_)



import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
tokenized = nltk.word_tokenize(text)
print(tokenized)
nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if(pos[:2] == 'NN')]
print (nouns)