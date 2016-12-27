# Yelp Reviews Sentiment Analysis
Below is a TL;DR rundown of the major components
## Introduction

The growing amount of information *encoded* in natural language flowing on the internet is placing more and more emphasis on meaningful information retrieval. For this project, a Maximum Entropy model was trained to classify a text review as positive, objective or negative. With the aid of outside resources such as [NLTK](http://www.nltk.org/), [Stanford NLP Part-of-Speech Taggers](http://nlp.stanford.edu/software/tagger.shtml), and [SentiWordNet](sentiwordnet.isti.cnr.it), an overall accuracy of 82% was achieved, using primarily bag-of-words features.

For more information, please see the [full project write up](Yelp-Reviews-Sentiment-Analysis.pdf)

## Data
The data set used throughout this project is obtained from Yelp, distributed as part of their [Yelp Dataset Challenge](https://www.yelp.com/dataset_challenge). The original dataset contains more than 2.5 million reviews. However, due to the time frame of this project, it was not feasible to utilize all of the existing reviews. Instead, among these, 40 000 reviews were selected to be the training set, 4 000 development set, and 4 000 test set, maintaining an 80% training and 20% validating ratio.

## Data Preprocessing
Preprocessing is done in two steps. First, all reviews are tokenized with NLTK's tokenizer `word_tokenize`. Second, all reviews are tagged with a Part-Of-Speech (POS) from the Penn Treebank tagset using the Stanford POS tagger. 
## Evalution
The following evaluation metrics are used to measure the performance of the models:
```
              tp                                              tp
Precision = -------                             Recall    = -------
            tp + fp                                         tp + fn

             2 * Precision * Recall                              tp + tn
F-Measure = ------------------------            Accuracy  = -----------------   
               Precision + Recall                           tp + tn + fp + fn  
```
Where `tp` are the true positives, `fp` - false positives, `tp` - true positives, and `tn` - true negatives. 

## Maximum Entropy
Maximum Entropy models are frequently used in natural language processing for classification tasks.Using the Natural Language Toolkit (NLTK) for Python, a Maximum Entropy classifier that has its weights chosen to maximize entropy while remaining empirically consistent with the training set was trained. The basis of the model is NLTK's `MaxentClassifier`, which is then wrapped inside another module `SentimentClassifier` specifically for this project.

## Feature Extraction
### What Matters
One important criteria in selecting which features out of all extracted features to encode is their [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) scores, or term-frequency versus inverse-document-frequency score. The usual formula of tf-idf measures the importance of a word to a particular document in a collection of documents, rewarding words that appear frequently in a document but infrequently in the collection of documents. However, as this is a classification problem, we are interested in features that are important to a particular classification label. Therefore, we want to reward features that appear relatively frequently in reviews of a particular label but relatively infrequently across all reviews. 

To do so, we introduce a slight twist to the usual tf-idf formula. For each classification feature `f` and label `l`,
```
tfidf(f, l) = tf(f, l) * idf(f, l)

               # of reviews labeled i that contains f
tf(f, l)    = ---------------------------------------
                      # of reviews labeled i

                       Total # of reviews
idf(f, l)   = log ----------------------------
                  # of reviews that contains f
```
### Bag of Words
A common feature used in sentiment analysis is bag-of-words. That is, to simplify the representation of each review into a bag-of-words that appeared in the review. Each word is a feature, which has a value of 1 if that word occurred in the review, and 0 otherwise. The same idea can be extended to not only unigrams but any number of grams. However, while a larger ngram can capture more contextual information, it risks losing generality and is too prone to overfitting. 
### Negation Detecion
Phrasal negation is an important syntactic feature that is almost impossible to detect with a simple bag-of-words approach. To detect negation, a set of negation words are hand selected. Whenever a negation word is detected, the sentiment polarity of the subsequent five words would be prepended a `NOT_` suffix, hence distinguishing between regular features and negated features. 
### SentiWordNet
[SentiWordNet](http://sentiwordnet.isti.cnr.it/) is a lexical resource for opinion mining based on [WordNet](http://wordnet.princeton.edu/). Each word in SentiWordNet belongs to a sentisynset that has three sentiment scores - positivity, objectivity and negativity - such that for any sentisynset:
```
positivity(sentisynset) + objectivity(sentisynset) + negativity(sentisynset) = 1
```
In this scoring system, it is possible for a word to have both a positive sentiment and a negative sentiment. But the sentiment of any word lies on a spectrum from 0 to 1, with 0 being the most negative and 1 being the most positive.

SentiWordNet is an invaluable resource to this classification problem. It identifies unigrams in the training set that has a sentiment score skewed to the classification it appeared it. We then proceed to identify the most significant label associated with any unigram. Therefore, the vocabulary of unigrams for which we will consider as features consist of the most significantly correlated classification label in addition to the unigrams themselves. In extracting features, the feature value is the SentiWordNet score of the most significant classificatio of the unigram if it appears in a review and 0 otherwise. This type of feature extraction is implemented in the module `FeatExtractor.SWNBoWFeatExtractor` and is used in final model.
### Performance on Test set
The final model, which utilizes SentiWordNet valued unigrams and negation, achieved an overall accuracy of 82%.
```
==> Confusion Matrix
                             PREDICTED
                  Positive   Objective    Negative
     ACTUAL    |-----------------------------------
    Positive   |    1374          217           75
    Objective  |     322          977          367
    Negative   |     104          274         1288

==> Evaluation Metrics
                  Positive   Objective    Negative
               |-----------------------------------
    Recalls    |    0.825       0.586       0.773
    Precisions |    0.763       0.666       0.745
    F-measures |    0.793       0.623       0.759
    Accuracies |    0.856       0.764       0.836

==> Average accuracy:  0.819
```

