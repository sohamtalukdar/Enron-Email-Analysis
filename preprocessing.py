df = None
def preprocess_data():

    global df 

    #Map folder names to label values
    labels = {
        "1": "Company Business, Strategy, etc.",
        "2": "Purely Personal",
        "3": "Personal but in professional context",
        "4": "Logistic Arrangements",
        "5": "Status arrangements",
        "6": "Document editing/checking",
        "7": "Empty message (due to missing attachment)",
        "8": "Empty message"
    }

    root_directory = 'enron_with_categories'
    rows = []

    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)
        if os.path.isdir(folder_path):
            label = labels[folder_name]
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    with open(os.path.join(folder_path, filename), 'r') as f:
                        data = f.read()
                        message_body = data.split("\n\n")[-1]
                        number = int(filename.split(".")[0])
                        rows.append({"#": number,"Label": label,"Message": message_body})

    df = pd.DataFrame(rows)
    # Drop rows with "Empty message (due to missing attachment)" or "Empty message" labels
    df = df[df['Label'] != 'Empty message (due to missing attachment)']
    df = df[df['Label'] != 'Empty message']

    # Drop rows with empty Message column
    df = df.dropna(subset=['Message'])
    df = df[df['Message'] != '']

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Clean the email message
    def clean_text(text):
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]|\d', '', text)
        text = text.lower()
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        tokens = word_tokenize(text)
        # Stem the words
        # Initialize the stemmer
        stemmer = SnowballStemmer("english")
        text = [stemmer.stem(word) for word in tokens]
        #text = [stemmer.stem(text) for text in text]
        # Remove punctuation and stop words
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return " ".join(tokens)
    df['Message'] = df['Message'].apply(lambda x: clean_text(x))

    return preprocess_data


    