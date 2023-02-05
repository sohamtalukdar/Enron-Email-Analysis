from libraries import *
from preprocessing import Preprocessor

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

preprocessor = Preprocessor()
preprocessor.preprocess_data()
df = preprocessor.get_dataframe()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Message'])

scaler = StandardScaler(with_mean=False)
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, df['Label'], test_size=0.2, random_state=0)

models = [
    MultinomialNB(),
    RandomForestClassifier(n_estimators=100, random_state=42),
    SVC(kernel='linear', C=1, random_state=42)
]

for model in models:
    print("\n" + model.__class__.__name__ + ":")
    evaluate_model(model, X_train, X_test, y_train, y_test)


