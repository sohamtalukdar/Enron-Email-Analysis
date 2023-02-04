from libraries import *
from preprocessing import Preprocessor

proc = Preprocessor()
proc.preprocess_data()
df_model = proc.get_dataframe()

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_model['Message'])

# Normalization
scaler = StandardScaler(with_mean=False)
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df_model['Label'], test_size=0.2, random_state=0)

def run_model(clf, name):
    # Train the model
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Naive Bayes
run_model(MultinomialNB(), "Naive Bayes")

# Random Forest
run_model(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")

# Support Vector Machine
run_model(SVC(kernel='linear', C=1, random_state=42), "Support Vector Machine")
