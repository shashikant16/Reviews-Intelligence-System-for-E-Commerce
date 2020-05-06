import os
from datetime import datetime
import re
os.chdir(r"E:\Projects\Pipeline_Code")

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

    def Extracting_Noun_Phrases(self,review):
        noun_phrases =  Noun_Phrase_Extraction.fetch_noun_phrases(review)
        return noun_phrases
    def Unigram_TF_IDF(self,text):
        unigram = Unigram_TFIDF.get_unigram_tfidf(text)
        return unigram
    def Bigram_TF_IDF(self,text):
        bigram =  Bigram_Trigram.get_bigram_tfidf(text)
        return bigram
        
    def Trigram_TF_IDF(self,text):
        trigram =  Bigram_Trigram.get_trigram_tfidf(text)
        return trigram
    def preprocessing_steps(self,index):
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
    def get_top_n_words(self,df, n=None):
        corpus = [self.preprocessing_steps(i) for i in range(len(df['unigrams']))]
        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in      
                      vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], 
                           reverse=True)
        return words_freq[:n]
    
    def Collocation_bigram_trigram(self,reviews):
        bigram_col,trigram_col =  Collocation_Function.get_cleaned_text(reviews)
        return bigram_col,trigram_col
    
    def process(self,text):
        final_word = text.strip()
        final_word = final_word.lower()
        return  final_word

    def search_space(self,df):
        print("inside")
        noun_phrases = list(df['noun_phrases'])
        noun_phrases = list(itertools.chain.from_iterable(noun_phrases))
        bigrams = list(df['bigrams'])
        for i in range(len(bigrams)):
            res = bigrams[i].strip('][').split(',') 
            res = [item.replace("'",'') for item in res]
            bigrams[i] = res
    
        bigrams = list(itertools.chain.from_iterable(bigrams))
        trigrams = list(df['trigrams'])    
        for i in range(len(trigrams)):
            res = trigrams[i].strip('][').split(',') 
            res = [item.replace("'",'') for item in res]
            trigrams[i] = res
        def process(text):
            final_word = text.strip()
            final_word = final_word.lower()
            return  final_word
        trigrams = list(itertools.chain.from_iterable(trigrams)) 
        trigram_col,bigram_col = self.Collocation_bigram_trigram(df['Review Comment'])
        trigram_col = [' '.join(i) for i in trigram_col]
        bigram_col = [' '.join(i) for i in bigram_col]
        search_space = noun_phrases+bigrams+trigrams+bigram_col+trigram_col
        search_space = list(filter(lambda x: x!="",search_space))
        search_space= [ self.process(text) for text in search_space]
        search_space= [ process(text) for text in search_space]
        return search_space
    
    def search_queries(self,word):
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
        antonyms=[]
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
        synonyms = synonyms+antonyms
        synonyms= list(set(synonyms))
        synonyms.append(variation)
        return synonyms
   
    
    def finding_related_phrases(self,search_query_list,search_space_list,word):
        searched_=[]
        newlist=[]
        keyword_related_phrases=[]
        for entity in search_query_list:
            r = re.compile(r'([a-zA-Z\s]*'+entity+')'or'('+entity+'[a-zA-Z\s]*)')
            newlist = list(filter(r.match, search_space_list)) 
            if len(newlist)>0:
                searched_.append(newlist)
        keyword_related_phrases = list(itertools.chain.from_iterable(searched_))
        self.keywords_and_keyphrases[word] = keyword_related_phrases
        return self.keywords_and_keyphrases
    def processing_dataframe(self,text):
        stop_words = set(stopwords.words("english"))
        new_words = ["Fu**","F***","F###"]
        stop_words = stop_words.union(new_words)
        text = re.sub('[^a-zA-Z]', ' ', str(text))
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
 
    def extend_dataframe(self,df):
        df['Review Comment'] = df['Review Comment'].apply(lambda x:self.processing_dataframe(x))
        df['noun_phrases'] = df['Review Comment'].apply(lambda x: self.Extracting_Noun_Phrases(x))
        df['unigrams'] = df['Review Comment'].apply(lambda x: self.Unigram_TF_IDF(x))
        df['bigrams'] = df['Review Comment'].apply(lambda x: self.Bigram_TF_IDF(x))
        df['trigrams'] = df['Review Comment'].apply(lambda x: self.Trigram_TF_IDF(x))
        df = df[df['Review Comment'] !=""]
        df= df.reset_index()
        df = df.drop('index',axis=1)
        return df
    
    def keyword_and_related_phrases(self,df):
        top_keywords = self.get_top_n_words(df, n=20)
        top_keywords = pd.DataFrame(top_keywords,columns=['Words','Frequency'])
        keywords= top_keywords.head(10)
        print(keywords)
        search_space_list = self.search_space(df)
        for word in list(keywords['Words'])+['return','store','competition','online','website','staff','choice','collection','recommend','cloth']:
            search_query_list = self.search_queries(word)
            keywords_and_keyphrases = self.finding_related_phrases(search_query_list,search_space_list,word)     
        return keywords_and_keyphrases

    


if __name__=="__main__":
    start_time  = datetime.now()
    df = pd.read_csv(r"filepath")
    df = df[['Review Title','Review Comment','Sentiment']]
    df = Review_extraction().extend_dataframe(df)
    result = Review_extraction().keyword_and_related_phrases(df)
    end_time = datetime.now() - start_time 
    

import json
j = json.dumps(result, indent=4)
f = open('keywords.json', 'w')
print(j,file= f)
f.close()