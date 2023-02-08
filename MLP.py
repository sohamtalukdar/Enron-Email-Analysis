from libraries import *
from preprocessing import Preprocessor,TextClassifier

# Create an instance of the Preprocessor class
preprocessor = Preprocessor()
preprocessor.preprocess_data()

# Create an instance of the TextClassifier class
tc = TextClassifier(preprocessor)

# Split the data into training and testing sets
tc.split_data()


# Define the hyperparameter search space
param_grid = {'hidden_layer_sizes': [(50,), (100,), (150,)],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'solver': ['lbfgs', 'sgd', 'adam'],
              'alpha': [0.0001, 0.001, 0.01]}

# Create the grid search object
gridSearch = GridSearchCV(MLPClassifier(max_iter=100), param_grid, cv=5)

# Fit the grid search to the training data
# grid_search.fit(train_vectors, tc.train_labels)
gridSearchCV = tc.grid_search(gridSearch)
# Get the best set of hyperparameters
best_params = gridSearchCV.best_params_

 # Train a classifier with the best hyperparameters
MLP =  MLPClassifier(max_iter=100, **best_params)

 # Evaluate the model

tc.train_and_evaluate(MLP)

# # Train a classifier with the best hyperparameters
# classifier = MLPClassifier(max_iter=100, **best_params)
# classifier.fit(train_vectors, train_labels)

#Predict the labels for the test data
#predictions = classifier.predict(test_vectors)

# # Evaluate the model using metrics such as accuracy, precision, recall, and F1-score
# accuracy = accuracy_score(test_labels, predictions)
# precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
# recall = recall_score(test_labels, predictions, average='weighted')

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
