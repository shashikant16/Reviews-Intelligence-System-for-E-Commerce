import nltk 
import re 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

def get_bigram_tfidf(txt1):

    txt1 = [txt1]

    # Stopword removal  
    stop_words = set(stopwords.words('english')) 
    your_list = ['####', '***','fucking'] 
    for i, line in enumerate(txt1): 
        txt1[i] = ' '.join([x for 
            x in nltk.word_tokenize(line) if 
            ( x not in stop_words ) and ( x not in your_list )])
    vectorizer = CountVectorizer(ngram_range =(2, 2))
    try:
        
        X1 = vectorizer.fit_transform(txt1)       ## Error  
        features = (vectorizer.get_feature_names()) 
        # Applying TFIDF 
        # You can still get n-grams here 
        vectorizer = TfidfVectorizer(ngram_range = (2, 2)) 
        X2 = vectorizer.fit_transform(txt1) #tf.fit_transform(smallcorp.split('\n')) 
        scores = (X2.toarray()) 
        # print("\n\nScores : \n", scores)

        # Getting top ranking features 
        sums = X2.sum(axis = 0) 
        data1 = [] 
        for col, term in enumerate(features): 
            data1.append( (term, sums[0, col] )) 
        ranking = pd.DataFrame(data1, columns = ['term', 'rank'])
        words = (ranking.sort_values('rank', ascending = False))
        words = words['term'].head(5)
        words = words.to_string(index=False)  ### Type Casting
        words = words.replace("\n",',')
        words = words.replace("\t",'')
        words = words.replace("  ",'')
    except:
        words = txt1
    return words
	
	


def get_trigram_tfidf(txt1):

    txt1 = [txt1]

    # Stopword removal  
    stop_words = set(stopwords.words('english')) 
    your_list = ['####', '***','fucking'] 
    for i, line in enumerate(txt1): 
        txt1[i] = ' '.join([x for 
            x in nltk.word_tokenize(line) if 
            ( x not in stop_words ) and ( x not in your_list )])
    vectorizer = CountVectorizer(ngram_range =(3, 3))
    try:
        
        X1 = vectorizer.fit_transform(txt1)       ## Error  
        features = (vectorizer.get_feature_names()) 
        # Applying TFIDF 
        # You can still get n-grams here 
        vectorizer = TfidfVectorizer(ngram_range = (3, 3)) 
        X2 = vectorizer.fit_transform(txt1) #tf.fit_transform(smallcorp.split('\n')) 
        scores = (X2.toarray()) 
        # print("\n\nScores : \n", scores)

        # Getting top ranking features 
        sums = X2.sum(axis = 0) 
        data1 = [] 
        for col, term in enumerate(features): 
            data1.append( (term, sums[0, col] )) 
        ranking = pd.DataFrame(data1, columns = ['term', 'rank'])
        words = (ranking.sort_values('rank', ascending = False))
        words = words['term'].head(5)
        words = words.to_string(index=False)  ### Type Casting
        words = words.replace("\n",',')
        words = words.replace("\t",'')
        words = words.replace("  ",'')
    except:
        words = txt1
    return words

