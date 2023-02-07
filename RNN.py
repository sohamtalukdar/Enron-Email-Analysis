### RNN GS CV

from libraries import *
from preprocessing import Preprocessor,TextClassifier
# Create an instance of the Preprocessor class

preprocessor = Preprocessor()
preprocessor.preprocess_data()
df = preprocessor.get_dataframe()

# Create an instance of the TextClassifier class
tc = TextClassifier(preprocessor)

# Split the data into training and testing sets
tc.split_data()

# Split data into train and test sets
X = df['Message']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Encode labels to integers
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

#Convert text data to numerical format using one-hot encoding
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)

#Define the model
def create_model(optimizer, units1, units2, units3):
  model = keras.Sequential([
  keras.layers.Dense(units1, activation='relu', input_shape=(1000,)),
  keras.layers.Dense(units2, activation='relu'),
  keras.layers.Dense(units3, activation='relu'),
  keras.layers.Dense(8, activation='softmax')
  ])
  
  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  return model

param_grid = {'optimizer': ['adam', 'rmsprop'],
'units1': [32, 64, 128],
'units2': [32, 64, 128],
'units3': [32, 64, 128]}

#Initialize the model
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

#Initialize GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

#Fit the model
grid_result = grid.fit(X_train, y_train)

#Print the best parameters and the corresponding score
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

