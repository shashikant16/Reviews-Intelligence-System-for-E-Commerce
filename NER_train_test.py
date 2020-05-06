# adding more features
#
# very fragile bit of code
#
# if aborted midway, plase copy paste the temp files into the original directory

from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import nltk
from nltk import word_tokenize, pos_tag
from sklearn.model_selection import train_test_split
import pycrfsuite
import os, os.path, sys
import glob
from xml.etree import ElementTree
import numpy as np
from sklearn.metrics import classification_report
import pycrfsuite
from seqeval.metrics import f1_score


# this function appends all annotated files
def append_annotations(files):
    xml_files = glob.glob(files + "/*.xml")
    xml_element_tree = None
    new_data = ""
    for xml_file in xml_files:
        # print(xml_file)
        data = ElementTree.parse(xml_file).getroot()
        # print ElementTree.tostring(data)
        temp = ElementTree.tostring(data)
        new_data += str(temp)
    return (new_data)


# this function removes special characters and punctuations
def remov_punct(withpunct):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    without_punct = ""
    char = 'nan'
    for char in withpunct:
        if char not in punctuations:
            without_punct = without_punct + char
    return (without_punct)


# functions for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]


def get_labels(doc):
    return [label for (token, postag, label) in doc]


def normalize_sentence_lengths(sents):
    if len(sents) < 1:
        return sents

    # merge all first
    merged_array = []
    for sent in sents:
        merged_array = merged_array + sent

    avg_sent_len = 20
    overlap = 5  # between cons sentences
    slide_value = avg_sent_len - overlap

    norm_sents = []

    start_index = 0
    end_index = min(avg_sent_len, len(merged_array))

    norm_sents.append(merged_array[start_index:end_index])

    while (end_index < len(merged_array)):
        start_index = start_index + slide_value
        end_index = min(end_index + slide_value, len(merged_array))
        norm_sents.append(merged_array[start_index:end_index])

    return norm_sents


def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]
    # Common features for all words. You may add more features here based on your custom use case
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]
    # Features for words that are not at the beginning of a document
    if i > 0:
        word1 = doc[i - 1][0]
        postag1 = doc[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    if i > 1:
        word1 = doc[i - 2][0]
        postag1 = doc[i - 2][1]
        features.extend([
            '-2:word.lower=' + word1.lower(),
            '-2:word.istitle=%s' % word1.istitle(),
            '-2:word.isupper=%s' % word1.isupper(),
            '-2:word.isdigit=%s' % word1.isdigit(),
            '-2:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS2')

    if i > 2:
        word1 = doc[i - 3][0]
        postag1 = doc[i - 3][1]
        features.extend([
            '-3:word.lower=' + word1.lower(),
            '-3:word.istitle=%s' % word1.istitle(),
            '-3:word.isupper=%s' % word1.isupper(),
            '-3:word.isdigit=%s' % word1.isdigit(),
            '-3:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS3')

    # Features for words that are not at the end of a document
    if i < len(doc) - 1:
        word1 = doc[i + 1][0]
        postag1 = doc[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    if i < len(doc) - 2:
        word1 = doc[i + 2][0]
        postag1 = doc[i + 2][1]
        features.extend([
            '+2:word.lower=' + word1.lower(),
            '+2:word.istitle=%s' % word1.istitle(),
            '+2:word.isupper=%s' % word1.isupper(),
            '+2:word.isdigit=%s' % word1.isdigit(),
            '+2:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS2')

    return features


input_dir = r"E:\Projects\TRC Learning\CRF '
temp_dir = 'temp'
output_dir = r'E:\Projects\TRC Learning\CRF '

input_files = os.listdir(input_dir)

newdata_pred_all = []
y_test_all = []

allxmlfiles = append_annotations(input_dir)
soup = bs(allxmlfiles, "html.parser")

# identify the tagged element
docs = []
doc_temp = []
sents = []

for d in soup.find_all("document"):
    # print(d)
    for wrd in d.contents:
        # print(wrd)
        tags = []
        NoneType = type(None)
        if isinstance(wrd.name, NoneType) == True:
            withoutpunct = remov_punct(wrd)
            temp = word_tokenize(withoutpunct)
            for token in temp:
                tags.append((token, 'NA'))
        else:
            # withoutpunct = remov_punct(wrd)
            # use the following in case of error
            try:
                withoutpunct = remov_punct(wrd)
            except Exception as e:
                print(wrd,' has exception ',e)
            temp = word_tokenize(withoutpunct)
            for token in temp:
                tags.append((token, wrd.name))
        sents = sents + tags
        # print('sentence length before normalization ',len(sents))
        # print(sents)
        doc_temp.append(sents)  # appends all the individual documents into one list
        sents = []
    doc_normalized = normalize_sentence_lengths(doc_temp)
    doc_temp = []
    for norm_sent in doc_normalized:
        # print('sentence length after normalization ',len(norm_sent))
        docs.append(norm_sent)

data = []

for i, doc in enumerate(docs):
    tokens = [t for t, label in doc]
    tagged = nltk.pos_tag(tokens)
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])

X = [extract_features(doc) for doc in data]
print(X)
y = [get_labels(doc) for doc in data]
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)#, random_state=42)

trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,

    # maximum number of iterations
    'max_iterations': 200,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

# Provide a file name as a parameter to the train function, such that
# the model will be saved to the file when training is finished
trainer.train('crf_v2.model')

tagger = pycrfsuite.Tagger()
tagger.open('crf_v2.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

# Create a mapping of labels to indices
labels = {"b-neg": 2,"i-pos":3,"b-pos":4, "i-neg": 1,"NA": 0}

# Convert the sequences of tags into a 1-dimensional array
predictions = np.array([labels[tag] for row in y_pred for tag in row])
truths = np.array([labels[tag] for row in y_test for tag in row])

# not printint f1 score because this might be incorrect for multiclass problems
# print('f1 score : ',f1_score(y_test, y_pred))

print(classification_report(
    truths, predictions,
    target_names=["NA","i-neg","b-neg","b-pos","i-pos"]))



i = 0

for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):

    print("%s (%s)" % (y, x))



