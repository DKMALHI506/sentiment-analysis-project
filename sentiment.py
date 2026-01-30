# sentiment.py
# Sentiment Analysis using NLTK and Naive Bayes

import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

# Step 0: Download movie reviews dataset
nltk.download("movie_reviews")

# Step 1: Prepare the dataset
# Each document is a tuple (words, category)
documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# Step 2: Define feature extractor
def extract_features(words):
    """
    Convert a list of words into a dictionary of features.
    Each word is a key and True is its value.
    """
    return {word: True for word in words}

# Step 3: Create feature sets
featuresets = [(extract_features(doc), category) for (doc, category) in documents]

# Step 4: Split into training and test sets
train_set = featuresets[:1600]
test_set = featuresets[1600:]

# Step 5: Train Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Step 6: Test classifier on test set
positive_reviews = 0
negative_reviews = 0

for review in test_set:
    sentiment = classifier.classify(review[0])
    if sentiment == 'pos':
        positive_reviews += 1
    else:
        negative_reviews += 1

print("Positive Reviews:", positive_reviews)
print("Negative Reviews:", negative_reviews)

# Step 7: Calculate accuracy
accuracy = nltk_accuracy(classifier, test_set)
print("Accuracy:", accuracy)

# Step 8: Test on your own sentence
input_text = "The movie was great and very entertaining!"
input_features = extract_features(input_text.split())
sentiment = classifier.classify(input_features)
print("Sentiment of input text:", sentiment)
