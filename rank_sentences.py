#!utf-8
import networkx as nx
import numpy as np
 
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
 
def textrank(document):
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(document)

    bow_matrix = CountVectorizer().fit_transform(sentences)
    normalized = TfidfTransformer().fit_transform(bow_matrix)
 
    similarity_graph = normalized * normalized.T
 
    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    return sorted(((scores[i],s) for i,s in enumerate(sentences)),
                  reverse=True)


#document =  "ebay is based in san jose. ebay was founded in 2015. I work at ebay."

#document = unicode(open('test_description.txt').read(),errors='replace')

document = unicode(open('/Users/ckhatri/Desktop/Content_Generation_Project/data/suman_data/cleaned_epid_1023944883.txt').read(),errors='replace')

#document = open('/Users/ckhatri/Desktop/Content_Generation_Project/cleaned_data_prp').readlines()[0]

scores = textrank(document)

for i in scores:
	print i[1].encode('utf-8')