from libraries import *
from preprocessing import Preprocessor,TextClassifier

# Create an instance of the Preprocessor class
preprocessor = Preprocessor()
preprocessor.preprocess_data()

# Create an instance of the TextClassifier class
tc = TextClassifier(preprocessor)

# Split the data into training and testing sets
tc.split_data()


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

    def grid_search(self, gridsearch):
        train_vectors = self.get_average_word_vectors(self.train_messages)
        gridsearch.fit(train_vectors, self.train_labels)
        return

    
    def train_and_evaluate(self, classifier):
        train_vectors = self.get_average_word_vectors(self.train_messages)
        classifier.fit(train_vectors, self.train_labels)

        test_vectors = self.get_average_word_vectors(self.test_messages)
        predictions = classifier.predict(test_vectors)

 

# Define the hyperparameter search space
param_grid = {'hidden_layer_sizes': [(50,), (100,), (150,)],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'solver': ['lbfgs', 'sgd', 'adam'],
              'alpha': [0.0001, 0.001, 0.01]}

# Create the grid search object
grid_search = GridSearchCV(MLPClassifier(max_iter=100), param_grid, cv=5)

# Fit the grid search to the training data
grid_search.fit(train_vectors, train_labels)

# Get the best set of hyperparameters
best_params = grid_search.best_params_

# Train a classifier with the best hyperparameters
classifier = MLPClassifier(max_iter=100, **best_params)
classifier.fit(train_vectors, train_labels)

# Predict the labels for the test data
predictions = classifier.predict(test_vectors)

# Evaluate the model using metrics such as accuracy, precision, recall, and F1-score
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
recall = recall_score(test_labels, predictions, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)