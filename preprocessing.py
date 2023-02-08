from libraries import *

class Preprocessor:

    def __init__(self):
        self.df = None  
        self.labels = {
            "1": "Company Business, Strategy, etc.",
            "2": "Purely Personal",
            "3": "Personal but in professional context",
            "4": "Logistic Arrangements",
            "5": "Status arrangements",
            "6": "Document editing/checking",
            "7": "Empty message (due to missing attachment)",
            "8": "Empty message"
        }
        self.root_directory = 'enron_with_categories'

    def preprocess_data(self):
        """
        Preprocesses the data by performing the following steps:
        1. Reads the text files and constructs a DataFrame with '#', 'Label', and 'Message' columns.
        2. Cleans the 'Message' column by removing punctuation, digits, email addresses, URLs, and stopwords.
        3. Stems the words in the 'Message' column.
        4. Drops the rows with empty 'Message' or 'Label' columns, or rows with "Empty message (due to missing attachment)" or "Empty message" labels.
        5. Removes duplicates.
        """
        rows = []

        for folder_name in os.listdir(self.root_directory):
            folder_path = os.path.join(self.root_directory, folder_name)
            if os.path.isdir(folder_path):
                label = self.labels[folder_name]
                for filename in os.listdir(folder_path):
                    if filename.endswith(".txt"):
                        with open(os.path.join(folder_path, filename), 'r') as f:
                            data = f.read()
                            message_body = data.split("\n\n")[-1]
                            number = int(filename.split(".")[0])
                            rows.append({"#": number, "Label": label, "Message": message_body})

        self.df = pd.DataFrame(rows)

        # Clean the email message
        self.df['Message'] = self.df['Message'].apply(lambda x: re.sub(r'[^\w\s]|\d', '', x).lower())
        self.df['Message'] = self.df['Message'].apply(lambda x: re.sub(r'\S+@\S+', '', x))
        self.df['Message'] = self.df['Message'].apply(lambda x: re.sub(r'http\S+', '', x))
        self.df['Message'] = self.df['Message'].apply(lambda x: " ".join([word for word in word_tokenize(x) if word.isalpha() and word not in set(stopwords.words("english"))]))
        self.df['Message'] = self.df['Message'].apply(lambda x: " ".join([SnowballStemmer("english").stem(word) for word in x.split()]))

        # Drop rows with "Empty message (due to missing attachment)" or "Empty message" labels
        self.df = self.df[self.df['Label'] != 'Empty message (due to missing attachment)']
        self.df = self.df[self.df['Label'] != 'Empty message']

        # Drop rows with empty Message column
        self.df = self.df.dropna(subset=['Message'])
        self.df = self.df[self.df['Message'] != '']

        # Remove duplicates
        self.df.drop_duplicates(inplace=True)
        
    def get_dataframe(self):

        return self.df




class TextClassifier:
    def __init__(self, preprocessor, sg=0, window=5, min_count=1, negative=10, seed=0):
        self.preprocessor = preprocessor
        self.df = preprocessor.get_dataframe()
        self.messages = self.df['Message'].apply(lambda x: x.split()).tolist()
        self.model = Word2Vec(self.messages, sg=sg, window=window, min_count=min_count, negative=negative, seed=seed)

    def split_data(self, test_size=0.2, random_state=0):
        self.train_df, self.test_df = train_test_split(self.df, test_size=test_size, random_state=random_state)
        self.train_labels = self.train_df['Label'].tolist()
        self.train_messages = self.train_df['Message'].apply(lambda x: x.split()).tolist()
        self.test_labels = self.test_df['Label'].tolist()
        self.test_messages = self.test_df['Message'].apply(lambda x: x.split()).tolist()

    def get_average_word_vectors(self, messages):
        vectors = np.zeros((len(messages), 100))
        for i, message in enumerate(messages):
            word_vectors = np.zeros((100,))
            for word in message:
                if word in self.model.wv:
                    word_vectors += self.model.wv[word]
            word_vectors = word_vectors / len(message)
            vectors[i] = word_vectors
        return vectors

    def grid_search(self, gridsearch):
        train_vectors = self.get_average_word_vectors(self.train_messages)
        gridsearch.fit(train_vectors, self.train_labels)
        return

    
    def train_and_evaluate(self, classifier):
        train_vectors = self.get_average_word_vectors(self.train_messages)
        classifier.fit(train_vectors, self.train_labels)

        test_vectors = self.get_average_word_vectors(self.test_messages)
        predictions = classifier.predict(test_vectors)

    


        accuracy = accuracy_score(self.test_labels, predictions)
        precision = precision_score(self.test_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(self.test_labels, predictions, average='weighted')
        f1 = f1_score(self.test_labels, predictions, average='weighted')
        

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)
    
    
    