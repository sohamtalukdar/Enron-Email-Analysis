from libraries import *
from preprocessing import Preprocessor

def vectorize_data(df_model):
    # Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_model['Message'])

    # Normalization
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)
    return X

def run_model(clf, X_train, y_train, y_test, name):
    # Train the model
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    preprocess = Preprocessor()
    preprocess.preprocess_data()
    display_df = preprocess.display()
    df_model = preprocess.get_dataframe()
    
    X = vectorize_data(df_model)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, df_model['Label'], test_size=0.2, random_state=0)

    # Naive Bayes
    run_model(MultinomialNB(), X_train, y_train, y_test, "Naive Bayes")

    # Random Forest
    run_model(RandomForestClassifier(n_estimators=100, random_state=42), X_train, y_train, y_test, "Random Forest")

    # Support Vector Machine
    run_model(SVC(kernel='linear', C=1, random_state=42), X_train, y_train, y_test, "Support Vector Machine")



