# sentiment-analysis-project
Using Python and the Natural Language Toolkit (NLTK) library

# Step 1 
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

# Download the Movie Reviews dataset
nltk.download("movie_reviews")

# Prepare data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Define a feature extractor function
def extract_features(words):
    return dict([(word, True) for word in words])

# Create feature sets
featuresets = [(extract_features(doc), category) for (doc, category) in documents]

# Split the data into training and testing sets
train_set = featuresets[:1600]
test_set = featuresets[1600:]

# Train the Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# step 2 

# Test the classifier
positive_reviews = 0
negative_reviews = 0
for review in test_set:
    sentiment = classifier.classify(review[0])
    if sentiment == 'pos':
        positive_reviews += 1
    if sentiment == 'neg':
        negative_reviews += 1

print("Positive Reviews:", positive_reviews)
print("Negative Reviews:", negative_reviews)

# step 3

# Calculate accuracy
accuracy = nltk_accuracy(classifier, test_set)
print("Accuracy:", accuracy)

# step 4

# Calculate accuracy
accuracy = nltk_accuracy(classifier, test_set)
print("Accuracy:", accuracy)
input_text = "The movie was great and very entertaining!"
input_features = extract_features(input_text.split())
sentiment = classifier.classify(input_features)
print("Sentiment:", sentiment)


