from libraries import *
from preprocessing import Preprocessor



from gensim.models import Word2Vec

class Word2VecCBOW:
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.df = preprocessor.get_dataframe()
        self.model = None

    def train_model(self, window_size=5, min_count=1, workers=4, iter=100):
        messages = self.df['Message'].tolist()
        messages = [message.split() for message in messages]
        self.model = Word2Vec(messages, window=window_size, min_count=min_count, workers=workers, iter=iter, sg=0)
        
    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = Word2Vec.load(path)

# Example usage
preprocessor = Preprocessor()
preprocessor.preprocess_data()
w2v = Word2VecCBOW(preprocessor)
w2v.train_model()
w2v.save_model("word2vec_cbow.model")
