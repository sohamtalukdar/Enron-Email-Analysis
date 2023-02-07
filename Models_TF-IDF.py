# Import required libraries and Preprocessor class
from libraries import *
from preprocessing import Preprocessor

# Evaluate a given machine learning model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Fit the model to training data
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Print accuracy score and classification report
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Preprocess data
preprocessor = Preprocessor()
preprocessor.preprocess_data()
df = preprocessor.get_dataframe()

# Convert text data to TF-IDF matrix
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Message'])

# Scale the matrix values
scaler = StandardScaler(with_mean=False)
X = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, df['Label'], test_size=0.2, random_state=0)

# List of machine learning models to evaluate
models = [
    MultinomialNB(),
    RandomForestClassifier(n_estimators=100, random_state=42),
    SVC(kernel='linear', C=1, random_state=42)
]

# Evaluate each model
for model in models:
    print("\n" + model.__class__.__name__ + ":")
    evaluate_model(model, X_train, X_test, y_train, y_test)
