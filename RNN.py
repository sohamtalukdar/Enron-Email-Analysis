### RNN GS CV

from libraries import *
from preprocessing import Preprocessor,TextClassifier
# Create an instance of the Preprocessor class
preprocessor = Preprocessor()
preprocessor.preprocess_data()
df = preprocessor.get_dataframe()

# Create an instance of the TextClassifier class
#tc = TextClassifier(preprocessor)

# Split the data into training and testing sets
#tc.split_data()

#Split data into train and test sets
# X = df['Message']
# y = df['Label']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #Encode labels to integers
# le = LabelEncoder()
# y_train = le.fit_transform(y_train)
# y_test = le.transform(y_test)

# #Convert text data to numerical format using one-hot encoding
# tokenizer = keras.preprocessing.text.Tokenizer(num_words=1000)
# tokenizer.fit_on_texts(X_train)
# X_train = tokenizer.texts_to_matrix(X_train)
# X_test = tokenizer.texts_to_matrix(X_test)

# #Define the model
# def create_model(optimizer, units1, units2, units3):
#   model = keras.Sequential([
#   keras.layers.Dense(units1, activation='relu', input_shape=(1000,)),
#   keras.layers.Dense(units2, activation='relu'),
#   keras.layers.Dense(units3, activation='relu'),
#   keras.layers.Dense(8, activation='softmax')
#   ])
  
#   model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#   return model

# param_grid = {'optimizer': ['adam', 'rmsprop'],
# 'units1': [32, 64, 128],
# 'units2': [32, 64, 128],
# 'units3': [32, 64, 128]}

# #Initialize the model
# model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

# #Initialize GridSearchCV
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

# #Fit the model
# grid_result = grid.fit(X_train, y_train)

# #Print the best parameters and the corresponding score
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Split data into train and test sets
X = df['Message'].to_numpy()
y = df['Label'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Encode labels to integers
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

#Convert text data to numerical format using one-hot encoding
tokenizer = keras.preprocessing.text.Tokenizer(num_words=1000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)

class TextClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

train_dataset = TextClassificationDataset(X_train, y_train)
test_dataset = TextClassificationDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Define the model
class TextClassificationModel(nn.Module):
    def __init__(self, units1, units2, units3):
        super(TextClassificationModel, self).__init__()
        self.fc1 = nn.Linear(1000, units1)
        self.fc2 = nn.Linear(units1, units2)
        self.fc3 = nn.Linear(units2, units3)
        self.fc4 = nn.Linear(units3, 8)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return self.softmax(x)

# Train the model

def train(model, train_loader, optimizer, criterion):
  model.train()
  for epoch in range(100):
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      y_pred = model(X_batch)
      loss = criterion(y_pred, y_batch)
      loss.backward()
      optimizer.step()

#Evaluate the model on the test set
model.eval()
with torch.no_grad():
  outputs = model(torch.tensor(X_test, dtype=torch.float32))
  _, predicted = torch.max(outputs.data, 1)
  correct = (predicted == torch.tensor(y_test)).sum().item()
  accuracy = correct / len(y_test)

print(f"Test accuracy: {accuracy:.2f}")