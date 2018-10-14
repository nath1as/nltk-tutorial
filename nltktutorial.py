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
