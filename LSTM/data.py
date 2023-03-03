import os
import torch
import numpy as np
import pandas as pd

from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


class TextLoader:
    def __init__(self, data_dir, vocab_size):
        self._vocab_size = vocab_size
        self.texts, self.labels = self.load_data(data_dir)
        self.processed_texts = self.token2id(self.texts)
        self.train_data,self.test_data = self.split_data(self.processed_texts,
                                                        self.labels)
        # self.train_loader = self.loader(self.train_data)

    def load_data(self, data_dir):
        filenames = os.listdir(data_dir)
        for file in filenames:
            if file.endswith('csv'):
                df = pd.read_csv(os.path.join(data_dir,file), 
                                encoding='latin-1')
                break
        # Remove NaN columns
        df_ = df.dropna(axis='columns')

        # Extractor Text, Label column
        for column in df_.keys():
            if 'ham' in df_[column].unique():
                label = np.where(df_[column]=='spam',1,0)
            else:
                text = df_[column]
        return text.tolist(), label.tolist() 

    def token2id(self, texts):
        tokenizer = Tokenizer(num_words=self._vocab_size)
        tokenizer.fit_on_texts(texts)
        X = tokenizer.texts_to_sequences(texts)
        X = pad_sequences(X)
        return X

    def split_data(self, texts, labels):
        X_train,X_test,Y_train,Y_test = train_test_split(texts,
                                                        labels,
                                                        test_size=0.1,
                                                        random_state=42)
        return (X_train, Y_train), (X_test, Y_test)


if __name__=="__main__":
    data_dir = 'data/archive'
    text_loader = TextLoader(data_dir)
    texts = text_loader.texts
    label = text_loader.labels
    print(label)