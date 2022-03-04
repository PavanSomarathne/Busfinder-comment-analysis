from distutils.log import debug
import random
import re;
import pickle
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from flask import Flask, jsonify, request #import objects from the Flask model
from flask_cors import CORS
from sqlalchemy import true

from flask import Flask

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

documents_f = open("app/pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("app/pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)
print(len(featuresets))

open_file = open('app/pickled_algos/originalnaivebayes5k.pickle', "rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open('app/pickled_algos/MNB_classifier5k.pickle', "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open('app/pickled_algos/BernoulliNB_classifier5k.pickle', "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open('app/pickled_algos/LogisticRegression_classifier5k.pickle', "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open('app/pickled_algos/LinearSVC_classifier5k.pickle', "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open('app/pickled_algos/SGDC_classifier5k.pickle', "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

givenpatterns = [(r'won\'t', 'will not'),
                (r'can\'t', 'cannot'),
                (r'i\'m', 'i am'),
                (r'ain\'t', 'is not'),
                (r'(\w+)\'ll', '\g<1> will'),
                (r'(\w+)n\'t', '\g<1> not'),
                (r'(\w+)\'ve', '\g<1> have'),
                (r'(\w+)\'s', '\g<1> is'),
                (r'(\w+)\'re', '\g<1> are'),
                (r'(\w+)\'d', '\g<1> would')
                    ]

def replace(text, patterns):
    for(row,rep) in patterns:
        regex = re.compile(row)
        text = regex.sub(rep,text)
#     print(text)
    return text

class RepeatReplacer(object):

    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def replace(self, word):
        if wordnet.synsets(word):
            return word

        repl_word = self.repeat_regexp.sub(self.repl, word)

        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word

voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


replacers = RepeatReplacer()

def newSentiment(text):
    results =  replacers.replace(text)
    feats = find_features(results)
    classifier.classify(feats)
    prob_dist = classifier.prob_classify(feats)
    return {'type' : voted_classifier.classify(feats) , 'rate'  :voted_classifier.confidence(feats)*100 ,'NB_Rate':prob_dist.prob(classifier.classify(feats))*100 } 

app = Flask(__name__)
CORS(app)

languages = [{'name' : 'JavaScript'}, {'name' : 'Python'}, {'name' : 'Ruby'}]

@app.route('/', methods=['GET'])
def index():
    return 'Welcome to career predictor'

@app.route('/langs', methods=['POST'])
def addOne():
    language = {'name' : newSentiment(request.json['name'])}
#     languages.append(language)
    return jsonify({'languages' : language})

@app.route('/lang', methods=['GET'])
def returnAll():
    return jsonify({'languages' : languages})

if __name__ == "__main__":
    app.run(debug=True)
