import os
import re
os.chdir(r"E:\Projects\NLP_Decathlon\Pipeline_Code")

######## Importing packages ##############

import pandas as pd 
# Libraries for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix
# Libraries for text Clustering
import math
import itertools
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize ,sent_tokenize

import Noun_Phrase_Extraction
import Unigram_TFIDF
import Bigram_Trigram
import Collocation_Function



class Review_extraction():
    keywords_and_keyphrases={}

    def Extracting_Noun_Phrases(review):
        noun_phrases =  Noun_Phrase_Extraction.fetch_noun_phrases(review)
        return noun_phrases
    def Unigram_TF_IDF(text):
        unigram = Unigram_TFIDF.get_unigram_tfidf(text)
        return unigram
    def Bigram_TF_IDF(text):
        bigram = Bigram_Trigram.get_bigram_tfidf(text)
        return bigram
    def Trigram_TF_IDF(text):
        trigram = Bigram_Trigram.get_trigram_tfidf(text)
        return trigram
    def preprocessing_steps(index):
        stop_words = set(stopwords.words("english"))
        new_words = ["Fu**","F***","F###"]
        stop_words = stop_words.union(new_words)
        text = re.sub('[^a-zA-Z]', ' ', str(df['unigrams'][index]))
        text = text.lower()
        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
        text=re.sub("(\\d|\\W)+"," ",text)
        text = text.split()
        ps=PorterStemmer()
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in  
            stop_words] 
        text = " ".join(text)
        return text
    def get_top_n_words(df, n=None):
        corpus = [preprocessing_steps(i) for i in range(len(df['unigrams']))]
        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in      
                      vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], 
                           reverse=True)
        return words_freq[:n]
    
    def Collocation_bigram_trigram(reviews):
        bigram_col,trigram_col =  Collocation_Function.get_cleaned_text(reviews)
        return bigram_col,trigram_col
    
    def process(text):
        final_word = text.strip()
        final_word = final_word.lower()
        return  final_word

    def search_space(df):
        noun_phrases = list(df['noun_phrases'])
        noun_phrases = list(itertools.chain.from_iterable(noun_phrases))
        bigrams = list(df['bigrams'])
        bigrams = list(itertools.chain.from_iterable(bigrams))
        trigrams = list(df['trigrams'])    
        trigrams = list(itertools.chain.from_iterable(trigrams)) 
        trigram_col,bigram_col = Collocation_bigram_trigram(df['Review Comment'])
        trigram_col = [' '.join(i) for i in trigram_col]
        bigram_col = [' '.join(i) for i in bigram_col]
        search_space = noun_phrases+bigrams+trigrams
        search_space = list(filter(lambda x: x!="",search_space))
        noun_phrases= [ process(text) for text in search_space]
       
        search_space = search_space+trigram_col+bigram_col
        return search_space
    def search_queries(word):
        ps = PorterStemmer() 
        stemmed_word=  ps.stem(word)
        print(stemmed_word)
        length = len(stemmed_word)
        sixty_of_length  = math.ceil(length*0.6)
        letter   = stemmed_word[:sixty_of_length]
        if len(letter)>2:
                variation = letter
        else:
            variation=word
        synonyms=[]
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())
        synonyms= list(set(synonyms))
        synonyms.append(variation)
        return synonyms
    
    def finding_related_phrases(search_query_list,search_space_list,word):
        searched_=[]
        newlist=[]
        entity_list=[]
        keyword_related_phrases=[]
        for entity in search_query_list:
            r = re.compile(entity+"[a-zA-Z]*")
            newlist = list(filter(r.match, search_space_list)) 
            x = re.compile("[a-zA-Z] "+entity)
            entity_list = list(filter(x.match,search_space_list))
            newlist = newlist+entity_list
            if len(newlist)>0:
                searched_.append(newlist)
        keyword_related_phrases = list(itertools.chain.from_iterable(searched_))
        keywords_and_keyphrases[word] = keyword_related_phrases
        return keywords_and_keyphrases

    def extend_dataframe(df):
        df['noun_phrases'] = df['Review Comment'].apply(lambda x: Extracting_Noun_Phrases(x))
        df['unigrams'] = df['Review Comment'].apply(lambda x:Unigram_TF_IDF(x))
        df['bigrams'] = df['Review Comment'].apply(lambda x: Bigram_TF_IDF(x))
        df['trigrams'] = df['Review Comment'].apply(lambda x: Trigram_TF_IDF(x))
        return df
    
    def keyword_and_related_phrases(df):
        top_keywords = get_top_n_words(df, n=20)
        top_keywords = pd.DataFrame(top_keywords,columns=['Words','Frequency'])
        keywords= top_keywords.head(10)
        search_space_list = search_space(df)
        for word in keywords['Words']:
            search_query_list = search_queries(word)
            keywords_and_keyphrases = finding_related_phrases(search_query_list,search_space_list,word)     
        return keywords_and_keyphrases

    
#Most frequently occuring words


if __name__=="__main__":
    df = pd.read_csv(r"E:\Projects\NLP_Decathlon\FW Extractions Mopinion Trustpilot\Review export product trustpilot.csv")
    df = df[['Review Title','Review Comment','Sentiment']]
    df = Review_extraction.extend_dataframe(df)
    result = Review_extraction.keyword_and_related_phrases(df)
    
