'''
SentimentClassifier.py for Yelp Reviews Sentiment Analysis Project

A Sentiment classifier that wraps around NLTK's MaxentClassfier 

By: Julie Jiang 
Dec 2016
COMP 150 NLP
'''
import pickle
import json
from nltk.classify import maxent
from collections import defaultdict
from FeatExtractor import BoWFeatExtractor, \
                             SyntacticChunksFeatExtractor

class SentimentClassifier:
    '''
    This is a sentiment classifier that is built upon NLTK's MaxentClassifier.

    This can be created with the classmethod ``train()`` from a set of 
    FeatExtractors and collection of datasets, or it can be initialized with a 
    JSON file of an NLTK's MaxentClassifier model saved from a previously 
    trained SentimentClassifier.

    Base: object

    '''
    def __init__(self, feature_extractors, model = None, saved_model = None):
        '''
        Initialize a SentimentClassifier object with list of feature_extractors
        (instances of FeatExtractor) and either model or path_to_model.

        Parameters: 
            feature_extractors(list(FeatExtractors)) - a list of instances of 
                `FeatExtractor`. 

            model(MaxentClassifier) - an instance of NLTK's MaxentClassifier

            saved_model(str) - the name of the pickle file that contains 
                an instance of NLTK's MaxentClassifier. 

        Note that while both `model` and `path_to_model` are optional arguments,
        one and only one of them must be provided in order for this class to be
        successfully initialized. Otherwise, ValueError will be raised.

        '''
        if model and not path_to_model:
            self.model = model
        elif path_to_model and not model:
            self.model = pickle.load(open(path_to_model, "rb"))
        else:
            raise ValueError
        self.POS, self.OBJ, self.NEG = "positive", "objective", "negative"
        self.labels = [self.POS, self.OBJ, self.NEG]
        self.feature_extractors = feature_extractors
    def evaluate(self, datasets):
        '''
        Evaluate the performance of this classifier on a collection of datasets,
        typically the development sets or the test sets. Outputs the results
        to terminal, including the confusion matrix and evaluation metrics (
        precisions, recalls, f-measures, accuracies, and average accuracy).

        Parameter:
            training_sets(list(str)) - a list of JSON file names for which the 
                this classifier will be built upon. Each line of each JSON file 
                is a review in the format of a JSON decodable object with the 
                two following fields:
                    1. "text"(list(list(str, str))) - a list of all the words
                        in this review with their POS tags
                    2. "label"(str) - the classification of the text
        '''
        #confusion matrix. The keys are classification labels. 
        confusion = defaultdict(lambda: defaultdict(float))
        for dataset in datasets:
            with open(dataset) as inFile
                for line in inFile:
                    datum = json.loads(line)
                    text, label = datum["text"], datum["label"]
                    # Feature vectors. Maps a feature to a feature value
                    feats = {} 
                    for extractor in self.feature_extractors:
                        feats.update(extractor.extract_fvec(text))
                    # If the maxent model was trained as a binary model, 
                    # the predicted label is objective if the probabilities of
                    # it being positive and negative are nearly equal.
                    if len(self.model.labels()) == 2:
                        label_probs = self.model.prob_classify(feats)
                        pos_label_prob = label_probs.prob(self.POS)
                        if pos_label_prob > 0.66 or pos_label_prob < 0.34:
                            pred_label = label_probs.max()
                        else:
                            pred_label = self.OBJ
                    # Else if the maxent model was trained as a trinary model, 
                    # the predicted label is simply the one that yields
                    # highest probability. 
                    else:
                        pred_label = self.model.classify(feats)
                    confusion[label][pred_label] += 1
        self.display_results(confusion)
    def compute_metrics(self, confusion):
        '''
        From the confusion matrix, compute the following evaluation metrics:
        precisions, recalls, f-measures, and accuracies. 

        Parameters: confusion(dict(dict)) - confusion matrix. The keys are
            classification labels. Let l1, l2 be labels. confusion[l1][l2] is 
            the number of times the model classified something as l1 when it is 
            really l2.

        Returns: metrics(dict(dict)) - evalution matrix. The keys of the outer
            dict are the names of the evaluation metric, and the keys of the
            inner dict are the classification label. 
        '''
        # Maps the name of a metric to its value for each classification label.
        # The keys of the nested dict are classification labels
        metrics = defaultdict(lambda: defaultdict(float))
        total = sum([confusion[l1][l2] \
                    for l1 in self.labels for l2 in self.labels])
        for l in self.labels:
            # True positives
            tp = float(confusion[l][l])
            # False positives
            fp = sum([confusion[l2][l] for l2 in self.labels]) - tp
            # False negatives
            fn = sum([confusion[l][l2] for l2 in self.labels]) - tp
            # True negatives
            tn = total - tp - fp - fn
            # If any ZeroDivisionError might occur, then fall back to 0
            metrics['Precisions'][l] = tp / (tp + fp) if tp + fp != 0 else 0
            metrics['Recalls'][l]    = tp / (tp + fn) if tp + fn != 0 else 0
            metrics['F-measures'][l] = 2 * tp / (2 * tp + fp + fn) \
                                       if tp + fp + fn != 0 else 0
            metrics['Accuracies'][l] = (tp + tn) / (tp + tn + fp + fn) \
                                        if tp + tn + fp + fn != 0 else 0
        return metrics
    def display_results(self, confusion):
        '''
        Print the confusion matrix, evaluation metrics and overall accuracy.

        Parameters: confusion(dict(dict)) - confusion matrix. The keys are
            classification labels. Let l1, l2 be labels. confusion[l1][l2] is 
            the number of times the model classified something as l1 when it is 
            really l2.
        '''
        metrics = self.compute_metrics(confusion) # get evaluation metrics
        # Print confusion matrix
        print("==> Confusion Matrix\n" + 29 * " " + "PREDICTED\n" + 16 * " "),
        print((3 * " ").join("{: >9}".format(l.title()) for l in self.labels))
        print(5 * " " + "ACTUAL    |" + 35 * "-")
        print("\n".join("    {: <11}|".format(l1.title()) + 
             ("     ".join("{: >8g}".format(confusion[l1][l2]) \
              for l2 in self.labels)) for l1 in self.labels))

        # Print evaluation metrics
        print("\n==> Evaluation Metrics\n" + 16 * " "),
        print("   ".join("{: >9}".format(l.title()) for l in self.labels) )
        print("               |" + 35 * "-")
        for m, values in metrics.iteritems():
            print(4 * " " + "{: <11}|".format(m)),
            print((4 * " ").join("{: >8.3f}".format(metrics[m][l]) \
                  for l in self.labels))

        # Report average accuracy
        print("\n==> Average accuracy: "),
        print("{:.3f}".format(sum(metrics['Accuracies'].values()) / 3))
        
    @classmethod
    def train(cls, feature_extractors, training_sets, max_iter = 100, 
              save = True, save_to_file = "maxent_model.pickle"):
        '''
        Train a SentimentClassifier and, in the process, an NLTK's 
        MaxEntClassifier, with a set of feature extractors and  a set 
        of training set. 

        Parameters:
            feature_extractors(list(FeatExtractor)) - a list of instances of 
                `FeatExtractor`. 

            training_sets(list(str)) - a list of JSON file names for which the 
                this classifier will be built upon. Each line of each JSON file 
                is a review in the format of a JSON decodable object with the 
                two following fields:
                    1. "text"(list(list(str, str))) - a list of all the words
                        in this review with their POS tags
                    2. "label"(str) - the classification of the text

            max_iter(int) - the maximum number of iterations in training an
                NLTK's MaxentClassifier model. Defaults to 100.

            save(bool) - if True, saves the MaxentClassifier model as a pickle
                file. Defaults to True.

            save_to_file(str) - if save, the name of the file to save to. 
                Defaults to `maxent_model.pickle`.

        Returns: an instance of the SentimentClassifier 
        Return type: SentimentClassfier
        '''
        # Build train toks for MaxentClassifier
        train_toks = []
        for extractor in feature_extractors:
            train_toks += extractor.build_train_toks(
                          training_sets, save = False)
        # Train a MaxentClassifier model
        model = maxent.MaxEntClassifier.train(train_toks, max_iter)
        if save:
            pickle.dump(model, open(save_to_file))
        return cls(feature_extractors, model = model)





