import nltk
from nltk.corpus import stopwords
import re
from nltk import word_tokenize,sent_tokenize

def fetch_noun_phrases(text):
    final_list =[]
    text_lst = sent_tokenize(text)

    def preproc(text):
        #Removing punctuations
        text_prep=re.sub(r'[^\w\s]','',text)
        text_prep=re.sub(r'\d+','',text_prep)
        text_prep=re.sub(r'\s+', ' ',text_prep)
        text_tokens = word_tokenize(text_prep)
        text_pos=nltk.pos_tag(text_tokens)
        return text_pos
    text_pos = list(map(preproc,text_lst))
    while([] in text_pos) : 
        text_pos.remove([]) 
    

    grammar = r"""

       NP:
        {<VB|VBD|VBG|VBP|VBZ|RB>*<IN>*<NBAR><IN|DT|TO>*<JJ.*|NN.*|VB|VBD|VBG|VBP|VBZ>*<CC>*<NN.*>} # 'in , and'
        {<VB.*|RB>*<NBAR>} # to Avoid the 'IN' ****
        {<JJ.*>+<RB>*<CC><RB>*<JJ.*>+}
        {<DT>+<NN.*>+<VB|VBD|VBG|VBP|VBZ|RB>+<DT>*<JJ.*>}   # Added Rule 3  
        {<RB.*>+<VB|VBD|VBG|VBP|VBZ>+} #1
        {<PRP.*>+<VB|VBD|VBG|VBP|VBZ>+<JJ.*>+}
        {<RB>+<NN.*>}
        {<NNP|NNPS><NN>?}  ##### Changed 9th dec
        {<VB|VBD|VBG|VBP|VBZ>+<RB.*>+}
        {<RP>*<IN>*<NNP|NNPS>}                         ## Added Rule
        {<VB|VBD|VBG|VBP|VBZ>+<IN>+<NN|VB> }
       # {<NBAR>}
           {<RB>+<PRP>*<VBP|VB|VBN>+}
           {<NNP|NNS>*<VBZ|VBP>+<RB|JJ>+}
           {<JJ>*<VBG>+}
           {<RB>+<JJ>*<NN>*}
           {<NNP>+<NN>*}
           {<JJ>+<IN>+<NN>?}
           {<VBN>*<RB>?}
           {<VB>+<DT>?<JJ>*<NN>+}
           {<DT>?<JJ>+<NN|NNS>+}
           """
    chunker = nltk.RegexpParser(grammar)
    
    def leaves(tree):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
            yield subtree.leaves()

   
    def acceptable_word(word):
        """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
        accepted = bool(2 <= len(word) <= 40)
        return accepted


    def get_terms(tree):
        for leaf in leaves(tree):
            term = [ w.lower() for w,t in leaf if acceptable_word(w) ]
            yield term
            
    for pos in text_pos:
        tree = chunker.parse(pos)
        terms = get_terms(tree)
        for term in terms:
            if len(term)>=2:
                final_list.append(' '.join(term))
        
    return final_list
    


