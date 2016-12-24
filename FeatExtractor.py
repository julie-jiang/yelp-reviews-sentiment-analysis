'''
FeatExtractor.py for Yelp Reviews Sentiment Analysis Project

Contains implementations for FeatExtractor, BoWFeatExtractor, 
SWNBoWFeatExtractor and SyntacticChunksFeatExtractor.

By: Julie Jiang 
Dec 2016
COMP 150 NLP
'''
import json
import pickle
from math import log
from operator import itemgetter
from collections import defaultdict
from nltk.corpus import sentiwordnet as swn

###############################################################################
#                         Feature Extractor Base class                        #
###############################################################################
class FeatExtractor(object):
    '''
    Extracts significant/relevant features from set of datasets and builds 
    training toks to be used in NLTK's MaxentClassifier.

    This can be created with the classmethod ``train()`` from a collection of
    datasets, or it can be initialized with a JSON file of vocab saved from
    a previously built FeatExtractor.
    
    Intended to be used via one of its derived classes: BoWFeatExtractor, 
    SWNBoWFeatExtractor and SyntacticChunksFeatExtractor.

    Base: object
    '''
    def __init__(self, vocab, saved_vocab):
        '''
        Initializes a FeatExtractor with either `vocab` or `saved_vocab`.
        Parameters:
            vocab(set) - a set of features 
            saved_vocab(str) - the name of a JSON file that contains a list of 
                               iterables of features under the field "vocab".
        '''
        # Must have one and only one of `vocab` and `saved_vocab`
        if vocab and not saved_vocab:
            self.vocab = vocab
        elif saved_vocab and not vocab:
            self.vocab = set([tuple(feat) for feat in \
                         json.load(open(saved_vocab))["vocab"]])
        else:
            raise ValueError
    def build_train_toks(self, datasets, save = True, save_to_file = None):
        '''
        Builds a list of training toks from the given dataset with `self.vocab`.
        These training toks are specifically built to be compatible with NLTK's
        MaxentClassifier.

        Parameters:
            datasets(list(str)) - a list of JSON file names for which the this 
                feature extractor will be built upon. Each line of each JSON 
                file is a review in the format of a JSON decodable object with 
                the two following fields:
                    1. "text"(list(list(str, str))) - a list of all the words
                        in this review with their POS tags
                    2. "label"(str) - the classification of the text

            save(bool) - if True, saves the training toks to a pickle 
                file. Defaults to True.

            save_to_file(str) - if save, the name of the file to save to. 
                Defaults to `name-of-FeatExtractor_feats_vocab.json`
        Returns: training toks compatible with NLTK's MaxentClassifier
        Return type: (list(tuple(dict, str))), where dict is a feature vector
            with keys = features and values = feature values
        '''
        train_toks = []
        for dataset in datasets:
            for line in open(dataset):
                datum = json.loads(line)
                text, label = datum["text"], datum["label"]
                fvec = self.extract_fvec(text)
                train_toks.append((fvec, label))
        if save:
            if save_to_file == None:
                save_to_file = "training_toks.pickle"
            pickle.dump(train_toks, open(save_to_file, "wb"))
        return train_toks
    @staticmethod
    def break_into_sents(text):
        '''
        Break a chunk of text into individual sentences divided by the period:
            [".", "."]
        Parameter: text(list(list(str, str))) - a list of all the words
                   in this review with their POS tags
        Yields: list(list(str, str)) - a list of all the words of a sentence
                in this review with their POS tags.
        Yield type: list(list(str, str))
        '''
        sent = []
        period = [".", "."]
        for word in text:
            sent.append(word)
            if word == period:
                yield sent
                sent = []
        yield sent
    @staticmethod
    def _it_matters(counts, label_counts, cutoff = 0.01):
        '''
        Compute the tfidf values of a feature and returns the best label for 
        associated with this feature if this feature is significant to any 
        classification, else None.

        The tfidf score of a feature f and a label i is defined to be
            tfidf(f, i) = tf(f, i) * idf(f, i)
        where
                        # of reviews labeled i that contains f
            tf(f, i) = ---------------------------------------
                                # of reviews labeled i
                                total # of reviews
            idf(f, i) = log -----------------------------
                             # of reviews that contains f
    
        Parameters: 
            counts(dict(key: label, value: int)) - the number of reviews of 
                this classification label that contains this feature.

            label_counts(dict(key: label, value: int)) - the total number of 
                reviews of this classification label

            cutoff(double) - if given, the max tfidf score has to be 
                greater than this value to be considered significant.

        Returns: the best label for associated with this feature if this 
                 feature is significant, else None.
        Return type: str or None. 
        '''
        # Total number of reviews that contains this feature
        count_sum = sum(counts.values()) 
        # Maps a classification label to the tf-idf score of (feature, label)
        scores = {}
        total = sum(label_counts.values()) # Total number of reviews
        for label, count in counts.iteritems():
            try:
                scores[label] = ((float(count) / label_counts[label]) * \
                                log(total/ count_sum))
            except ZeroDivisionError:
                scores[label] = 0
        best_label, best_score = max(scores.iteritems(), key = itemgetter(1))
        return best_label if best_score > cutoff else None

###############################################################################
#                         Feature Extractor Base class                        #
###############################################################################
class BoWFeatExtractor(FeatExtractor):
    '''
    Contains Bag-of-Words (abbreviated BoW) features.
    This can be created with the classmethod ``train()`` from a collection of
    datasets, or it can be initialized with a JSON file of vocab saved from
    a previously built BoWFeatExtractor.
    Base class: FeatExtractor
    '''
    def __init__(self, vocab = None, saved_vocab = None, negate = True):
        '''
        Parameters:
            Must provide one and only of `vocab` and `saved_vocab`! 

            vocab(set(features)) - contains a set of BoW features.

            saved_vocab(str) - the name of a JSON file that contains a list of 
                iterables of features under the field "vocab".

            negate(bool) - if True, will turn on negation detection.
        '''
        super(BoWFeatExtractor, self).__init__(vocab, saved_vocab)
        self.negate = negate
        self.negations = set(["no", "not", "rather", "hardly", "n't"])
        self.negation_scope = 5
    def extract_fvec(self, text):
        '''
        Extract the BoW feature vector of this text.
        Parameters: text(list(list(str, str))) - a list of words with their 
                    POS tags
        Returns: a BoW feature vector of this text.
        Return type: dict(key = feature, value = 1 or 0)
        '''
        fvec = defaultdict(int)
        num_negate = 0
        for word, tag in text:
            if self.negate and word in self.negations:
                num_negate = self.negation_scope + 1
            elif (word, tag) in self.vocab:
                if num_negate > 0:
                    fvec[("NOT_" + word, tag)] = 1
                else:
                    fvec[(word, tag)] = 1
            num_negate -= 1
        return fvec


    @classmethod
    def train(cls, datasets, cutoff = 0.01, negate = True, save = True, 
              save_to_file = "bow_feats_vocab.json"):
        '''
        Trains a BoW Feature Extractor from the most significant word + POS tag 
        combos it finds in the given collection of datasets. A word + POS tag 
        is considered significant if 
            1) it has a nonzero sentiment according to SentiWordNet;
            2) and it has a tf-idf score higher than `cutoff` for at least one 
               classification label.

        Parameters: 
            datasets(list(str)) - a list of JSON file names for which the this 
                feature extractor will be built upon. Each line of each JSON 
                file is a review in the format of a JSON decodable object with 
                the two following fields:
                    1. "text"(list(list(str, str))) - a list of all the words
                        in this review with their POS tags
                    2. "label"(str) - the classification of the text
                    
            cutoff(double) - a feature will only be included if it has a tf-idf
                score that that is at least `cutoff`. Defaults to 0.01.

            negate(bool) -- if True, turns on degation detection. Defaults
                to True.

            save(bool) -- if True, saves the extracted features to a JSON 
                file under the field "vocab". Defaults to True.

            save_to_file(str) - if save, the name of the file to save to. 
                Defaults to "bow_features_vocab.json"

        Returns: An instance of the BoWFeatExtractor
        Return type: BoWFeatExtractor        
        '''
        # Maps a Penn TreeBank POS Tag to a WordNet POS tag
        wn_tag_maps = {"JJ": "a", "JJR": "a", "JJS": "a", "NN": "n", 
                       "NNS": "n", "NNP": "n", "NNPS": "n", "RB": "r", 
                       "RBR": "r", "RBS": "r", "VB": "v", "VBD": "v", 
                       "VBN": "v", "VBP": "v", "VBZ": "v"}
        # `word_counts` maps a lowercase word + its POS tag to the number of 
        # documents it appeared in under a classification label.
        # The keys of the nested dicts are the classification labels. 
        words_counts = defaultdict(lambda: defaultdict(int))
        # Maps a classification label to the number of times a review with that
        # label appeared.
        label_counts = defaultdict(int)
        vocab = set() # The resulting set of extract features

        # Step 1: extract words that are relevant according to sentiwordnet
        for dataset in datasets:
            for line in open(dataset):
                datum = json.loads(line)
                seen = set()
                label = datum["label"]
                label_counts[label] += 1
                for word, tag in datum["text"]:
                    try:
                        # The corresponding WordNet tag 
                        wn_tag = wn_tag_maps[tag] 
                        sss = swn.senti_synset(\
                                      word.lower() + "." + wn_tag + ".01")
                        # If this word + tag combo has a positive or negative
                        # negative sentiment according to Sentiwordnet
                        if max([sss.pos_score(), sss.neg_score()]) > 0:
                            seen.add((word.lower(), tag))
                    # Either the POS tag doesn't have a corresponding Wordnet
                    # tag or this combo of words and tags doesn't have a 
                    # SentiWordNet entry. In either the case, this word + tag
                    # combination is insignificant to our purpose of 
                    # information extraction.
                    except: 
                        pass
                 # For each word + tag combo that has a positive or negative 
                 # sentiment that appeared at least once in this document
                for word, tag in seen:
                    words_counts[(word.lower(), tag)][label] += 1
       
         # From these, extract words that are also significant in terms of 
        # their tf-idf scores
        for words, counts in words_counts.iteritems():
            if FeatExtractor._it_matters(counts, label_counts, cutoff):
                vocab.add(tuple(words))

        if save:
            json.dump({"vocab": [v for v in vocab]}, open(save_to_file, "w"))
        return cls(vocab, negate)

###############################################################################
#                 SentiWordNet Bag of Words Feature Extractor                 #
###############################################################################
class SWNBoWFeatExtractor(BoWFeatExtractor):
    '''
    Contains SentiWordNet valued Bag-of-Words (abbreviated SWN BoW) features.
    This can be created with the classmethod ``train()`` from a collection of
    datasets, or it can be initialized with a JSON file of vocab saved from
    a previously built SWNBoWFeatExtractor.
    Base class: FeatExtractor
    '''
    def __init__(self, vocab = None, saved_vocab = None, negate = True):
        super(SWNBoWFeatExtractor, self).__init__(vocab, saved_vocab, negate)
        # Maps a Penn TreeBank POS Tag to a WordNet POS tag
        self.wn_tag_maps = {"JJ": "a", "JJR": "a", "JJS": "a", "NN": "n", 
                            "NNS": "n", "NNP": "n", "NNPS": "n", "RB": "r", 
                            "RBR": "r", "RBS": "r", "VB": "v", "VBD": "v", 
                            "VBN": "v", "VBP": "v", "VBZ": "v"}
        # Maps a classification label to the corresponding SentiWordNet 
        # function name 
        self.swn_func_map = {'positive': 'pos_score', 
                             'objective': 'obj_score', 
                             'negative': 'neg_score'}
        vocab = {}
        for word, tag, label in self.vocab:
            vocab[(word, tag)] = label
        self.vocab = vocab
        # Must have one and only one of `vocab` and `saved_vocab`
    def extract_fvec(self, text):
        '''
        Extract the SWN BoW feature vector of this text.
        Parameter: text(list(list(str, str))) - a list of words with their 
            POS tags. 
        Returns: a SWN BoW feature vector of this text.
        Return type: dict(key = (word, tag), 
                          value = SentiWordNet score or 0)))
        '''

        fvec = defaultdict(float)
        num_negate = 0
        for word, tag in text:
            # If this word is a negation word
            if self.negate and word in self.negations: 
                num_negate = self.negation_scope + 1
            elif (word, tag) in self.vocab:
                # Obtain the SentiWordNet score for the set of word, tag and 
                # label.
                label = self.vocab[(word, tag)]
                sentiset = word + "." + self.wn_tag_maps[tag] + ".01"
                sentiscore = getattr(swn.senti_synset(sentiset), 
                                     self.swn_func_map[label])()
                if num_negate > 0: # If we are within scope of negation
                    fvec[("NOT_" + word, tag)] = - sentiscore
                else:
                    fvec[(word, tag)] = sentiscore
            num_negate -= 1
        return fvec
    def build_train_toks(self, datasets, save = True, save_to_file = None):
        '''
        Builds a list of training toks from the given dataset with `self.vocab`.
        These training toks are specifically built to be compatible with NLTK's
        Maxent Classifier.

        Parameters:
            datasets(list(str)) - a list of JSON file names for which the this 
                feature extractor will be built upon. Each line of each JSON 
                file is a review in the format of a JSON decodable object with 
                the two following fields:
                    1. "text"(list(list(str, str))) - a list of all the words
                        in this review with their POS tags
                    2. "label"(str) - the classification of the text

            save(bool) - if True, saves the training toks to a pickle 
                file. Defaults to True.

            save_to_file(str) - if save, the name of the file to save to. 
                Defaults to `swn_bow_feats_vocab.json`

        Returns: training toks compatible with NLTK's MaxentClassifier
        Return type: (list(tuple(dict, str))), where dict is a feature vector
            with keys = features and values = feature values
        '''
        train_toks = []
        for dataset in datasets:
            with open(dataset) as inFile:
                for line in inFile:
                    datum = json.loads(line)
                    text, label = datum["text"], datum["label"]
                    fvec = self.extract_fvec(text)
                    train_toks.append((fvec, label))
        if save:
            if save_to_file == None:
                save_to_file = "training_toks.pickle"
            pickle.dump(train_toks, open(save_to_file, "wb"))
        return train_toks

    @classmethod
    def train(cls, datasets, cutoff = 0.01, negate = True, save = True, 
              save_to_file = "swn_bow_feats_vocab.json"):
        '''
        Trains a BoW Feature Extractor from the most significant word + POS tag 
        combos it finds in the given collection of datasets. A word + POS tag 
        is considered significant if 
            1) it has a nonzero sentiment according to SentiWordNet;
            2) and it has a tf-idf score that is at least `cutoff` for at least 
               one classification label.

        Parameters: 
            datasets(list(str)) - a list of JSON file names for which the this 
                feature extractor will be built upon. Each line of each JSON 
                file is a review in the format of a JSON decodable object with 
                the two following fields:
                    1. "text"(list(list(str, str))) - a list of all the words
                        in this review with their POS tags
                    2. "label"(str) - the classification of the text
                    
            cutoff(double) - a feature will only be included if it has a tf-idf
                score that that is at least `cutoff`. Defaults to 0.01.

            negate(bool) -- if True, turns on degation detection. Defaults
                to True.

            save(bool) -- if True, saves the extracted features to a JSON 
                file under the field "vocab". Defaults to True.

            save_to_file(str) - if save, the name of the file to save to. 
                Defaults to "bow_features_vocab.json"

        Returns: An instance of the SWNBoWFeatExtractor
        Return type: SWNBoWFeatExtractor        
        '''
        # Maps a Penn TreeBank POS Tag to a WordNet POS tag
        wn_tag_maps = {"JJ": "a", "JJR": "a", "JJS": "a", "NN": "n", 
                       "NNS": "n", "NNP": "n", "NNPS": "n", "RB": "r", 
                       "RBR": "r", "RBS": "r", "VB": "v", "VBD": "v", 
                       "VBN": "v", "VBP": "v", "VBZ": "v"}
        # Maps a classification label to the corresponding SentiWordNet 
        # function name 
        func_map = {'positive': 'pos_score', 'negative': 'neg_score', 
                    'objective': 'obj_score'}
        # Maps a lowercase word + its POS tag to the number of reviews under a 
        # classification label that contains it
        # The keys of the nested dicts are the classification labels. 
        words_counts = defaultdict(lambda: defaultdict(int))
        # Maps a classification to the number of reviews with that label
        label_counts = defaultdict(int)
        vocab = set()

        for dataset in datasets:
            with open(dataset) as inFile:
                for line in inFile:
                    datum = json.loads(line)
                    seen = set()
                    label = datum["label"]
                    label_counts[label] += 1
                    for word, tag in datum["text"]:
                        try:
                            # The corresponding WordNet tag 
                            wn_tag = wn_tag_maps[tag] 
                            sss = swn.senti_synset(
                                  word.lower() + "." + wn_tag + ".01")
                            # If this word + tag combo has a positive or 
                            # negative sentiment according to Sentiwordnet
                            if getattr(sss, func_map[label])() > 0.33:
                                seen.add((word.lower(), tag))
                        # Either the POS tag doesn't have a corresponding 
                        # Wordnet tag or this combo of words and tags doesn't 
                        # have a SentiWordNet entry. In either the case, this 
                        # word + tag combination is insignificant to our 
                        # purposes of information extraction.
                        except: 
                            pass
                     # For each word + tag combo that has a positive or negative 
                     # sentiment that appeared at least once in this document
                    for word, tag in seen:
                        words_counts[(word.lower(), tag)][label] += 1
        # From these, extract words that are also significant in terms of 
        # their tf-idf scores
        for (word, tag), counts in words_counts.iteritems():
            best_label = FeatExtractor._it_matters(counts, label_counts, cutoff)
            if best_label:
                vocab.add((tuple([word, tag, best_label])))

        if save:
            json.dump({"vocab": [v for v in vocab]}, open(save_to_file, "w"))
        return cls(vocab, negate)

###############################################################################
#                       Syntactic Chunks Feature Extractor                 #
###############################################################################

class SyntacticChunksFeatExtractor(FeatExtractor):
    '''
    Extracts and builds syntactic chunk features.

    This can be created with the classmethod ``train()`` from a collection of
    datasets, or it can be initialized with a JSON file of vocab saved from
    a previously built SyntacticChunksFeatExtractor.

    Base class: FeatExtractor
    '''
    def __init__(self, vocab = None, saved_vocab = None):
        '''
        Parameters:
            Must provide one and only of `vocab` and `saved_vocab`! 

            vocab(set(features)) - contains a set of BoW features.

            saved_vocab(str) - the name of a JSON file that contains a list of 
                iterables of features under the field "vocab".
        '''
        super(SyntacticChunksFeatExtractor, self).__init__(vocab, saved_vocab)
        # n is the size of a syntactic chunks.
        for v in self.vocab:
            self.n = len(v)
            break

    def extract_fvec(self, text):
        '''
        Extract a syntactic chunk feature vector from the given text.
        Parameter: text(list(list(str, str))) - a list of words with their 
                   POS tags
        Returns: a syntactic chunk feature vector of this text.
        Return type: dict(key = syntactic chunk, value = 1 or 0))
        '''
        # Maps a syntactic chunk to its value. 1 if it is present, 0 otherwise.
        fvec = defaultdict(int)
        extracted_chunks = set()
        for sent in FeatExtractor.break_into_sents(text):
            # For each seen syntactic chunk, if it exists in the vocab,
            # then added it to the feature vector. 
            for i in range(len(sent) - self.n + 1):
                chunk = tuple([sent[i + j][1] for j in range(self.n)])
                if chunk in self.vocab:
                    fvec[chunk] = 1
        return fvec
    
    @classmethod
    def train(cls, datasets, n = 3, cutoff = 0.01, save = True, 
              save_to_file = "syntactic_chunks_feats_vocab.json"):
        '''
        Trains a Syntactic Chunks Feature Extractor from the most significant
        syntactic chunks it finds in the given collection of datasets. A 
        syntactic chunk is significant if tf-idf score higher than `cutoff`
        for at least one classification label

        Parameters: 
            datasets(list(str)) - a list of JSON file names for which the this 
                feature extractor will be built upon. Each line of each JSON 
                file is a review in the format of a JSON decodable object with 
                the two following fields:
                    1. "text"(list(list(str, str))) - a list of all the words
                        in this review with their POS tags;
                    2. "label"(str) - the classification of the text.

            n(int) - the number of grams to consider.
                    
            cutoff(double) - a feature will only be included if it has a tf-idf
                score that that is at least `cutoff`. Defaults to 0.01.

            save(bool) -- if True, saves the extracted features to a JSON 
                file under the field "vocab". Defaults to True.

            save_to_file(str) - if save, the name of the file to save to. 
                Defaults to "bow_features_vocab.json"

        Returns: An instance of the SyntacticChunkFeatExtractor
        Return type: SyntacticChunkFeatExtractor        
        '''
        # Maps a syntactic chunk to the number of reviewss of a under a 
        # classification label that contains it.
        # The keys of the nested dicts are the classification labels. 
        chunk_counts= defaultdict(lambda: defaultdict(int))
        # Maps a classification to the number of reviews with that label
        label_counts = defaultdict(int)
        vocab = set()
        for dataset in datasets:
            with open(dataset) as inFile:
                for line in inFile:
                    total += 1
                    extracted_chunks = set()
                    datum = json.loads(line)
                    label = datum["label"]
                    label_counts[label] += 1
                    for sent in FeatExtractor.break_into_sents(datum["text"]):
                        # For each ngram syntactic chunk
                        for i in range(len(sent) - n + 1):
                            chunk = tuple([sent[i + j][1] for j in range(n)])
                            extracted_chunks.add(chunk)
                    # For each chunk that is seen at least once in this document
                    for chunk in extracted_chunks:
                        chunk_counts[chunk][label] += 1
        # Add the syntactic chunks that are also significant in terms of 
        # their tf-idf scores to `vocab`
        for chunk, counts in chunk_counts.iteritems():
            if FeatExtractor._it_matters(counts, label_counts, cutoff):
                vocab.add(chunk)
        if save:
            json.dump({"vocab": [v for v in vocab]}, open(save_to_file, "w"))
        return cls(vocab)


