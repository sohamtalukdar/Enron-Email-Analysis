from libraries import *
from preprocessing import Preprocessor

# Create an instance of the Preprocessor class
preprocessor = Preprocessor()
preprocessor.preprocess_data()
df = preprocessor.get_dataframe()


# Get the messages as a list of lists of words
# Convert the messages column of the dataframe to a list of lists of words
messages = df['Message'].apply(lambda x: x.split()).tolist()

# Train the CBOW word2vec model
# Initialize the CBOW Word2Vec model with the following parameters:
# sg=1 for using the skip-gram architecture
# window=5 for the size of the sliding window 
# min_count=1 to consider words that appear only once
# negative=10 for the number of negative samples to be used during training
# seed=0 for the random seed
model = Word2Vec(messages, sg=0, window=5, min_count=1, negative=10, seed=0)

# Split the data into train and test sets
# Split the dataframe into a train set and a test set, using a test size of 0.2 and a random seed of 0
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Get the train and test labels and messages
# Get the list of labels and messages from the train and test sets, respectively
train_labels = train_df['Label'].tolist()
train_messages = train_df['Message'].apply(lambda x: x.split()).tolist()
test_labels = test_df['Label'].tolist()
test_messages = test_df['Message'].apply(lambda x: x.split()).tolist()

# Convert the messages to average word vectors
# Initialize an array to store the average word vectors for the train messages
train_vectors = np.zeros((len(train_messages), 100))

# Loop through the train messages, calculate the average word vectors for each message
for i, message in enumerate(train_messages):
    # Initialize an array to store the word vectors for each word in the current message
    vectors = np.zeros((100,))
    
    # Loop through each word in the current message
    for word in message:
        # If the word is in the vocabulary of the word2vec model
        if word in model.wv:
            # Add the word vector to the sum of word vectors for the current message
            vectors += model.wv[word]
    
    # Divide the sum of word vectors by the number of words in the message to get the average word vector
    vectors = vectors / len(message)
    # Store the average word vector for the current message in the train_vectors array
    train_vectors[i] = vectors

# Repeat the process for the test messages
test_vectors = np.zeros((len(test_messages), 100))
for i, message in enumerate(test_messages):
    vectors = np.zeros((100,))
    for word in message:
        if word in model.wv:
            vectors += model.wv[word]
    vectors = vectors / len(message)
    test_vectors[i] = vectors


# Define the hyperparameter search space
param_grid = {'hidden_layer_sizes': [(50,), (100,), (150,)],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'solver': ['lbfgs', 'sgd', 'adam'],
              'alpha': [0.0001, 0.001, 0.01]}

# Create the grid search object
gridSearch = GridSearchCV(MLPClassifier(max_iter=100), param_grid, cv=5)

# Fit the grid search to the training data
gridSearch.fit(train_vectors, tc.train_labels)
#gridSearchCV = tc.grid_search(gridSearch)
# Get the best set of hyperparameters
best_params = gridSearch.best_params_

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
