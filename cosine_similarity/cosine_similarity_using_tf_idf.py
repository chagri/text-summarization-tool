
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA
from nltk.stem.porter import *

stemmer = PorterStemmer()

def cosine_similarity(string1, string2):

	#string1 = unicode(string1.lower(), errors='replace')
	#string2 = unicode(string2.lower(), errors='replace')

	string1 = ' '.join([stemmer.stem(i) for i in string1.split(' ')])
	string2 = ' '.join([stemmer.stem(i) for i in string2.split(' ')])

	train_set = [string1]
	test_set = [string2]
	#train_set = ["The sky is blue.", "The sun is bright."] #Documents
	#test_set = ["moon light white."] #Query
	stopWords = stopwords.words('english')

	vectorizer = CountVectorizer(stop_words = stopWords)
	#print vectorizer
	transformer = TfidfTransformer()
	#print transformer

	trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
	testVectorizerArray = vectorizer.transform(test_set).toarray()
	#print 'Fit Vectorizer to train set', trainVectorizerArray
	#print 'Transform Vectorizer to test set', testVectorizerArray
	cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)

	for vector in trainVectorizerArray:
	    #print vector
	    for testV in testVectorizerArray:
	        #print testV
	        cosine = cx(vector, testV)
	        #print cosine

	return cosine

	#transformer.fit(trainVectorizerArray)
	#print
	#print transformer.transform(trainVectorizerArray).toarray()

	#transformer.fit(testVectorizerArray)
	#print 
#tfidf = transformer.transform(testVectorizerArray)
#print tfidf.todense()


#str1= "The sky is blue. The sun is bright." #Documents
#str2= "sun bright." #Query
#print cosine_similarity(str1,str2)
