# Reviews-Intelligence-System for E-Commerce
This repo contains the code to perform a 360-degree review analysis using.

•	Keyphrase extraction from User Reviews  
•	Sentiment Analysis on Reviews Using Deep learning technique.  
•	Review Clustering with ELMO Embedding.  
•	Review Analysis and visualization.  

The whole process will help to understand the business logic and business understanding.


# Summary of the project

Key Phrase Extraction using Grammer rule  
Deep Learning-Based Sentiment classifier  
ELMO embedding based clustering.  

# Usage
Noun_Phrase_Extraction.py - Creates key phrases from reviews.  
Unigram_TFIDF.py - It Generates the Top Unigram, top TFIDF Frequency count, wise.  
Bigram_Trigram_TFIDF.py - It Generates the Top Bigram_Trigram, top Frequency count, wise.  
Collocation_Function.py - Generating for Collocation of words.  
Organized_code_v1.py -  A-Class Function with all the functionalities.
NER_train_test.py - Sentiment Classifier using Conditional Random Field (CRF). 
User Input KWD.ipynb- this is used for where the keywords for a particular word Are found.  
User Input KWD Chart.ipynb-  this is used for where the positive neg neutral classes are plotted.  
Flow is the user is prompted to enter a word and that word is passed to User Input Kwd file for finding all the words related to that input and then they are all passed to Chart generation for finding the pos neg neu classes and the chart is returned.
