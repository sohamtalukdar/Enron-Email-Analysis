from libraries import *
from preprocessing import Preprocessor,TextClassifier

# Create an instance of the Preprocessor class
preprocessor = Preprocessor()
preprocessor.preprocess_data()

# Create an instance of the TextClassifier class
tc = TextClassifier(preprocessor)

# Split the data into training and testing sets
tc.split_data()

# Train and evaluate SVC model
print("Support Vector Classifier")
svc = SVC(kernel='linear', C=1)
tc.train_and_evaluate(svc)

# Train and evaluate GradientBoostingClassifier model
print("Gradient Boosting Classifier")
gbc = GradientBoostingClassifier()
tc.train_and_evaluate(gbc)
