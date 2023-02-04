from libraries import *
from preprocessing import Preprocessor

preprocessor = Preprocessor()
preprocessor.preprocess_data()
df = preprocessor.get_dataframe()


# Get the messages as a list of lists of words
messages = df['Message'].apply(lambda x: x.split()).tolist()

# Train the CBOW word2vec model
model = Word2Vec(messages, sg=0, window=5, min_count=1, negative=10, seed=0)

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Get the train and test labels and messages
train_labels = train_df['Label'].tolist()
train_messages = train_df['Message'].apply(lambda x: x.split()).tolist()
test_labels = test_df['Label'].tolist()
test_messages = test_df['Message'].apply(lambda x: x.split()).tolist()

# Convert the messages to average word vectors
train_vectors = np.zeros((len(train_messages), 100))
for i, message in enumerate(train_messages):
    vectors = np.zeros((100,))
    for word in message:
        if word in model.wv:
            vectors += model.wv[word]
    vectors = vectors / len(message)
    train_vectors[i] = vectors

test_vectors = np.zeros((len(test_messages), 100))
for i, message in enumerate(test_messages):
    vectors = np.zeros((100,))
    for word in message:
        if word in model.wv:
            vectors += model.wv[word]
    vectors = vectors / len(message)
    test_vectors[i] = vectors

# Train a classifier on the train data
classifier = SVC(kernel='linear', C=1)
classifier.fit(train_vectors, train_labels)

# Predict the labels for the test data
predictions = classifier.predict(test_vectors)

# Evaluate the model using metrics such as accuracy, precision, recall, and F1-score
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
recall = recall_score(test_labels, predictions, average='weighted')
f1_score = f1_score(test_labels, predictions, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)

