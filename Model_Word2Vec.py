from libraries import *
from preprocessing import Preprocessor

# class TextClassifier:
#     def __init__(self, preprocessor, sg=0, window=5, min_count=1, negative=10, seed=0):
#         self.preprocessor = preprocessor
#         self.preprocessor.preprocess_data()
#         self.df = preprocessor.get_dataframe()
#         self.messages = self.df['Message'].apply(lambda x: x.split()).tolist()
#         self.model = Word2Vec(self.messages, sg=sg, window=window, min_count=min_count, negative=negative, seed=seed)

#     def split_data(self, test_size=0.2, random_state=0):
#         self.train_df, self.test_df = train_test_split(self.df, test_size=test_size, random_state=random_state)
#         self.train_labels = self.train_df['Label'].tolist()
#         self.train_messages = self.train_df['Message'].apply(lambda x: x.split()).tolist()
#         self.test_labels = self.test_df['Label'].tolist()
#         self.test_messages = self.test_df['Message'].apply(lambda x: x.split()).tolist()

#     def get_average_word_vectors(self, messages):
#         vectors = np.zeros((len(messages), 100))
#         for i, message in enumerate(messages):
#             word_vectors = np.zeros((100,))
#             for word in message:
#                 if word in self.model.wv:
#                     word_vectors += self.model.wv[word]
#             word_vectors = word_vectors / len(message)
#             vectors[i] = word_vectors
#         return vectors


    

#     def train(self, kernel='linear', C=1):
#         self.train_vectors = self.get_average_word_vectors(self.train_messages)
#         self.classifier = SVC(kernel=kernel, C=C)
#         self.classifier.fit(self.train_vectors, self.train_labels)

#     def evaluate(self):
#         self.test_vectors = self.get_average_word_vectors(self.test_messages)
#         self.predictions = self.classifier.predict(self.test_vectors)
#         self.accuracy = accuracy_score(self.test_labels, self.predictions)
#         self.precision = precision_score(self.test_labels, self.predictions, average='weighted', zero_division=0)
#         self.recall = recall_score(self.test_labels, self.predictions, average='weighted')
#         self.f1_score = f1_score(self.test_labels, self.predictions, average='weighted')

# # Create an instance of the Preprocessor class
# preprocessor = Preprocessor()
# preprocessor.preprocess_data()

# # Create an instance of the TextClassifier class
# tc = TextClassifier(preprocessor)

# # Split the data into training and testing sets
# tc.split_data()

# # Train the Word2Vec model
# tc.train()

# # Evaluate the Word2Vec model
# tc.evaluate()

# # Print the Evaluated results
# print("Accuracy:", tc.accuracy)
# print("Precision:", tc.precision)
# print("Recall:", tc.recall)
# print("F1-Score:", tc.f1_score)


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from gensim.models import Word2Vec

from preprocessing import Preprocessor

class TextClassifier:
    def __init__(self, preprocessor, sg=0, window=5, min_count=1, negative=10, seed=0):
        self.preprocessor = preprocessor
        self.df = preprocessor.get_dataframe()
        self.messages = self.df['Message'].apply(lambda x: x.split()).tolist()
        self.model = Word2Vec(self.messages, sg=sg, window=window, min_count=min_count, negative=negative, seed=seed)

    def split_data(self, test_size=0.2, random_state=0):
        self.train_df, self.test_df = train_test_split(self.df, test_size=test_size, random_state=random_state)
        self.train_labels = self.train_df['Label'].tolist()
        self.train_messages = self.train_df['Message'].apply(lambda x: x.split()).tolist()
        self.test_labels = self.test_df['Label'].tolist()
        self.test_messages = self.test_df['Message'].apply(lambda x: x.split()).tolist()

    def get_average_word_vectors(self, messages):
        vectors = np.zeros((len(messages), 100))
        for i, message in enumerate(messages):
            word_vectors = np.zeros((100,))
            for word in message:
                if word in self.model.wv:
                    word_vectors += self.model.wv[word]
            word_vectors = word_vectors / len(message)
            vectors[i] = word_vectors
        return vectors
    
    def train_and_evaluate(self, classifier):
        train_vectors = self.get_average_word_vectors(self.train_messages)
        classifier.fit(train_vectors, self.train_labels)

        test_vectors = self.get_average_word_vectors(self.test_messages)
        predictions = classifier.predict(test_vectors)

        accuracy = accuracy_score(self.test_labels, predictions)
        precision = precision_score(self.test_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(self.test_labels, predictions, average='weighted')
        f1 = f1_score(self.test_labels, predictions, average='weighted')
        
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", F1)


# Train and evaluate SVC model
svc = SVC(kernel='linear', C=1)
train_and_evaluate_model(svc, train_vectors, train_labels, test_vectors, test_labels)

# Train and evaluate GradientBoostingClassifier model
gbc = GradientBoostingClassifier()
train_and_evaluate_model(gbc, train_vectors, train_labels, test_vectors, test_labels)
