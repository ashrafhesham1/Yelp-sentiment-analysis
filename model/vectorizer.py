import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader

import spacy
import nltk
import numpy as np
import pandas as pd
from collections import Counter
import json
import os
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import string


class Vocabulary:
    '''the class process the text and extract vocabulary'''
    
    def __init__(self, token_to_idx=None, add_unk=True, unk_token='<UNK>'):
        '''
        Args:
            - token_to_idx(dict): a dictionary that maps tokens to indices
            - add_unk(bool): a flag that indicates whether to add an unknown token or not
            - unk_token(str): the defalut unknown token - default: <UNK>
        '''
        if token_to_idx is None:
            token_to_idx = {}
            
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
        
        self._add_unk = add_unk
        self._unk_token = unk_token
        
        self.unk_idx = -1
        if self._add_unk:
            self.unk_idx = self.add_token(self._unk_token)
    
    def to_serializable(self):
        '''return the class data as a dictionary that can be serializable'''
        return {
            'token_to_idx': self._token_to_idx,
            'add_unk': self._add_unk,
            'unk_token': self._unk_token
        }
    
    @classmethod
    def from_serializable(cls, serializable):
        '''instantiates the class from serializable'''
        return cls(**serializable)
    
    def add_token(self, token):
        ''' 
        add new token to the vocabulary
        
        Args:
            - token(str)
        
        Returns:
            - idx(int): the index of the new added token
        '''
        if token in self._token_to_idx:
            return self._token_to_idx[token]
        
        idx = len(self._token_to_idx)
        self._token_to_idx[token] = idx
        self._idx_to_token[idx] = token
        
        return idx
    
    def add_many(self, tokens):
        '''
        add a list of tokens to the vocabulary
        
        Args:
            - tokens(list): the tokens list
        
        Returns:
            - indices(list): a list of indices corresponding to the tokens
        '''
        return [self.add_token(token) for token in tokens]
    
    def lookup_token(self, token):
        '''
        retrieve the index associated with a token 
        
        Args:
            - token(str)
        
        Returns:
            - idx(int)
        '''
        if self.unk_idx >= 0:
            return self._token_to_idx.get(token, self.unk_idx)
        
        try:
            return self._token_to_idx[token]
        except:
            raise KeyError(f'Unknown Token: {token} ')
    
    def lookup_index(self, idx):
        '''
        retrieve the token associated with an index
        
        Args:
            - idx(int)
        
        Returns:
            - token(str)
        '''
        try:
            return self._idx_to_token[idx]
        except:
            raise KeyError(f'Unknow index: {idx}')
    
    def __str__(self):
        return f'<Vocabulary(size={len(self)})>'
    
    def __len__(self):
        return len(self._token_to_idx)
    

class Vectorizer(object):
    '''this class encapsulate the vocabulry and prepare it for the models'''
    
    def __init__(self, x_vocab, y_vocab, encoding='one_hot'):
        '''
        Args:
            - x_vocab(Vocabulary): maps words to integers 
            - y_vocab(Vocabulary): maps class labels to integers
            - encoding(str): the type of encoding - options: 'one_hot' - 'tf_idf' - default: 'one_hot'
        '''
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab
        
        self.encoding = encoding
        
    def vectorize(self, x):
        '''
        given an observation create a vectorized representation 
        
        Args:
            - x(str): the observation
        '''
        
        if self.encoding == 'one_hot':
            
            rep = np.zeros(len(self.x_vocab), dtype=np.float)
            for token in x.split():
                if token not in string.punctuation:
                    rep[self.x_vocab.lookup_token(token)] = 1
        
            return rep
    
    @classmethod
    def from_dataframe(cls, df, x_col, y_col, cut_off=25):
        '''
        Inistintiate the class from a dataframe
        
        Args:
            - df(pandas.DataFrame): the dataset
            - x_col(str): the name of the column that contains the observations
            - y_col(str): the name of the column that contains the labels
            - cut_off(int): the frequency-based filtering parameter - default = 25
        
        Returns:
            - (Vectorizer)
        '''
        
        x_vocab, y_vocab = Vocabulary(add_unk=True), Vocabulary(add_unk=False)
        
        # add y tokens
        for y in sorted(set(df[y_col])):
            y_vocab.add_token(y)
        
        # add x tokens if token count > cut off
        word_counter = Counter()
        for x in df[x_col]:
            for word in x.split():
                word_counter[word] += 1
        
        for word, count in word_counter.items():
            if count >= cut_off:
                x_vocab.add_token(word)
        
        return cls(x_vocab, y_vocab)
    
    @classmethod
    def from_serializable(cls, serializable):
        """Instantiate a ReviewVectorizer from a serializable dictionary
        
        Args:
            serializable (dict): the serializable dictionary
        Returns:
            an instance of the ReviewVectorizer class
        """
        x_vocab = Vocabulary.from_serializable(serializable['x_vocab'])
        y_vocab = Vocabulary.from_serializable(serializable['y_vocab'])
        
        return cls(x_vocab, y_vocab, serializable['encoding'])
    
    def to_serializable(self):
        """
        Create the serializable dictionary for caching
        
        Returns:
            serializable (dict): the serializable dictionary
        """
            
        return {
            'x_vocab': self.x_vocab.to_serializable(),
            'y_vocab': self.y_vocab.to_serializable(),
            'encoding': self.encoding
        }

