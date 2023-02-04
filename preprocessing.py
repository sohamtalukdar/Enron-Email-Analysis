def preprocess_data():

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
    df['Message'] = df['Message'].apply(lambda x: re.sub(r'[^\w\s]|\d', '', x).lower())
    df['Message'] = df['Message'].apply(lambda x: re.sub(r'\S+@\S+', '', x))
    df['Message'] = df['Message'].apply(lambda x: re.sub(r'http\S+', '', x))
    df['Message'] = df['Message'].apply(lambda x: " ".join([word for word in word_tokenize(x) if word.isalpha() and word not in set(stopwords.words("english"))]))
    df['Message'] = df['Message'].apply(lambda x: " ".join([SnowballStemmer("english").stem(word) for word in x.split()]))


    return df