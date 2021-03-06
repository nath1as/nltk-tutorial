import os
import nltk
import nltk.corpus

from nltk.corpus import brown
hamlet = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')

# for word in hamlet[:500]: 
#     print(word, sep=' ', end=' ')


AI = "When it comes to artificial intelligence (AI), perhaps very few people can claim they fathered a huge part of it. One such man is Jürgen Schmidhuber. Schmidhuber is considered the father of very deep learning, and the pioneer of deep learning neural networks. In fact, he built the foundations for many of the AI systems we find in our smartphones today.  If anyone can predict how far AI will go in the next couple years, it’s him.  During a talk at WIRED2016, Schmidhuber presented the future of AI as something beyond just taking over jobs. In 2050 there will be trillions of self-replicating robot factories on the asteroid belt, he told the audience. A few million years later, AI will colonize the galaxy.  Schmidhuber believes AI will play a crucial role in the way we will gather resources, most abundantly found in space. Orbital robot factories will be (un)manned by AI, capable of self-replication and space exploration. These AI will be scientists, he says, and in a few million years, will naturally explore the galaxy out of curiosity, setting their own goals. Humans are not going to play a big role there, but that’s ok, says Schmidhuber."

from nltk.tokenize import word_tokenize
AI_tokens = word_tokenize(AI)

# print(brown.words())
# print(nltk.corpus.gutenberg.fileids())
# print(hamlet)
# print(AI_tokens)
# print(len(AI_tokens))

from nltk.probability import FreqDist
fdist = FreqDist()

for word in AI_tokens:
    fdist[word.lower()] +=1

print(word, fdist)

fdist_top10 = fdist.most_common(10)
print(fdist_top10)

from nltk.tokenize import blankline_tokenize
AI_blank = blankline_tokenize(AI)
print((AI_blank[0]))

string = "All the laws of matter are those which our mind must fabricate, and the only laws of mind are fabricated by matter."

quotes_token = nltk.wordpunct_tokenize(string)
quotes_bigrams = list(nltk.bigrams(quotes_token))
print(quotes_bigrams)

# stemming : normalize words into its base form or root form

from nltk.stem import PorterStemmer

pst = PorterStemmer()

print(pst.stem("having"))

words_to_stem = ['give', 'giving', 'given', 'gave']

for words in words_to_stem:
    print(words + ":" + pst.stem(words))


from nltk.stem import LancasterStemmer

lst = LancasterStemmer()

for words in words_to_stem:
    print(words + ":" + lst.stem(words))




from nltk.stem import SnowballStemmer

sbst = SnowballStemmer( 'english' )

for words in words_to_stem:
    print(words + ":" + sbst.stem(words))


# lemmatization: group together different inflected forms
#of a word, called Lemma, similar to  stemming with a proper word as an output

from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
word_lem = WordNetLemmatizer()

word_lem.lemmatize('corpora')

for words in words_to_stem:
    print(words + " : " + word_lem.lemmatize(words))

from nltk.corpus import stopwords

print(stopwords.words('english'))

import re
punctuaction = re.compile(r'[.,?!.:;()|0-9]')

sent = "Hegel was right, you know."
sent_tokens = word_tokenize(sent)

for token in sent_tokens:
    print(nltk.pos_tag([token]))

sent2 = "Kant considered marriage a contract."
sent_tokens2 = word_tokenize(sent2)

for token in sent_tokens2:
    print(nltk.pos_tag([token]))

# named entity recognition

from nltk import ne_chunk

NE_sent = " The US Presiedent stays in the WHITE HOUSE"

NE_tokens = word_tokenize(NE_sent)
NE_tags = nltk.pos_tag(NE_tokens)

NE_NER = ne_chunk(NE_tags)
print(NE_NER)


#
new = "The big cat ate the little mouse who was after fresh cheese."
new_tokens = nltk.pos_tag(word_tokenize(new))


grammar_np = r"NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar_np)

chunk_result = chunk_parser.parse(new_tokens)
print(chunk_result)

