from nltk.corpus import stopwords
import pickle
import numpy as np
class WordEmbedding:
    
    def __init__(self, word_embedding_path, unknown_strategy='default'):
        self.stopwords = set(stopwords.words('english'))
        self.embed_model = self.get_glove_dict(word_embedding_path)
        self.embed_dim = len(self.embed_model['car'])
        print("Dimension of a word vector: {}".format(len(self.embed_model['car'])))
        
        if unknown_strategy == 'default':
            unknown = np.zeros(self.embed_dim)
            unknown[1] = 1
            self.embed_model['<UNK>'] = unknown
        elif unknown_strategy == 'random':
            np.random.seed(7)
            self.embed_model['<UNK>'] = np.random.uniform(-0.01, 0.01, 300).astype("float32")
        elif unknown_strategy == 'zeros':
            self.embed_model['<UNK>'] = np.zeros(self.embed_dim)
        else:
            self.embed_model['<UNK>'] = None
            
    def load_pickle(self,file_path):
        with open(file_path,'rb') as f:
            return pickle.load(f)
            
    def get_glove_dict(self,word_embedding_path):
        return self.load_pickle(word_embedding_path)
    
    def get_vector(self, word):
        if word in self.embed_model.keys():
            return self.embed_model[word]
        elif word.lower() in self.embed_model.keys():
            return self.embed_model[word.lower()]
        else:
            return self.embed_model['<UNK>']
        