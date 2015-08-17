import sys
sys.path.insert(0, '/Users/ckhatri/Desktop/Content_Generation_Project/data/suman_data/cosine_similarity')
sys.path.insert(0,'/Users/ckhatri/Desktop/Content_Generation_Project')
import copy

from cosine_similarity import compare_texts
from tf_idf_similarity import cosine_similarity
from word_2_vec_cosine_similarity import cosine_similarity as cs
import pickle
import sys
import numpy as np
#text1 = "chandra khatri"
#text2 = "khushbu khatri"

f_apple_vectors = open('/Users/ckhatri/Desktop/Content_Generation_Project/apple_word_vecs')
word_vectors = pickle.load(f_apple_vectors)

#print word_vectors

#sys.exit()



def get_nonsimilar_part(string1, string2):
    if len(string1) > len(string2):
        pass
    else:
        pass


raw_data = open('cooker.txt').readlines()

raw_data_cleaned = open('cleaned_word2vec_cooker.txt','w')


#raw_data = ['ebay is based in San jose', 'ebay near san jose', 'fundamentals of past']
raw_data2 = copy.deepcopy(raw_data)

print len(raw_data)

inclusion_set = set()
exclusion_set = set()

line_vectors = {}

for i in raw_data:
    line = i.strip('.')
    line = line.strip().split(' ')

    this_vec = []
    for word in line:
        if word in word_vectors:
            this_vec.append(word_vectors[word])
        else:
            this_vec.append([0.0]*50)

    line_vectors[i.strip()] = np.sum(this_vec, axis=0)




for i in raw_data:
    inclusion_set.add(i.strip())

seen_set = set()

for i in raw_data:

    for j in range(len(raw_data)):

        line1 = i.strip()
        line2 = raw_data[j].strip()

        vec1 = line_vectors[line1]
        vec2 = line_vectors[line2]

        #print len(vec1) 
        #print len(vec2)
        
        score = cs(vec1, vec2)
        
        print score
        
        if (line1,line2) not in seen_set and not line1 == line2: 
            if score > 0.9 and score <= 1:
                #print line1, line2
                if len(line1) < len(line2):

                    #inclusion_set.add(line1)
                    exclusion_set.add(line2)
                else:
                    #inclusion_set.add(line2)
                    exclusion_set.add(line1)



            
            seen_set.add((line1,line2))
            seen_set.add((line2,line1)) 

        


for line in exclusion_set:
    line = line.strip()
    inclusion_set.remove(line)



print len(raw_data), len(inclusion_set)

#print '.\n'.join(list(inclusion_set))
raw_data_cleaned.write('.\n'.join(list(inclusion_set)))



#compare_texts(text1, text2) 