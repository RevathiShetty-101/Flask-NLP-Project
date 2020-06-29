import string
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

class Prepare:
    
    def __init__(self, df):
        self.stopwords = stopwords.words('english') 
        self.df = df
        self.tokenizer = Tokenizer()
        self.premise = self.valid_tokens(df['premise'].values)
        self.hypothesis = self.valid_tokens(df['hypothesis'].values)
        self.tokenizer.fit_on_texts(self.premise + self.hypothesis)
        self.vocab_size = len(self.tokenizer.word_index)+1
        self.vocab = self.tokenizer.word_index.items()
        self.max_len = 100
        self.label_map = {'correct':0, 'incorrect':1, 'contradictory':2}
        self.premise, self.hypothesis = self.prep_df()
        
    def valid_tokens(self,texts):
        valid = []
        for text in texts:
            valid_text = [token for token in str(text).split() if token.lower() not in self.stopwords ]
            valid.append(valid_text)
        return valid
                
        
    def prep_df(self):
        premise = self.tokenizer.texts_to_sequences(self.premise) 
        hypothesis = self.tokenizer.texts_to_sequences(self.hypothesis)
        #max1 = len(max(premise, key=len))
        #max2 = len(max(hypothesis, key=len))
        #self.max_len = max(max1, max2)
        premise = pad_sequences(premise, maxlen = self.max_len, padding='post')
        hypothesis = pad_sequences(hypothesis, maxlen = self.max_len, padding='post')
        return premise,hypothesis   