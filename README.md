Yet to automate it. 
The idea is to first remove all the junk sentences then remove the duplicates using remove_duplicates code given some similarity score. It uses cosine similarity with stemming, tf-idf etc.
Then Finally ranking the remaining sentences based on Page Rank algorithm. 
Whatever is left out is the summary.
