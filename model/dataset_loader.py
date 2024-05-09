from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from model.vectorizer import Vectorizer

class ReviewDataset(Dataset):
    ''' this class encapsulate the reviews dataset '''
    def __init__(self, review_df, vectorizer=None, encoding='one_hot', x_col='review', y_col='rating', cut_off = 25):
        '''
        Args:
            - revies_df(pandas.df): the dataset
            - vectorizer(Vectorizer): a vectorizer pre-instantiated from the dataset
                -- if vectotizer is set to None the object will instantiate a new one
            - encoding(str): the type of encoding (used only when the vectorizer is set to None)
                -- options: 'one_hot' - 'tf_idf' - default: 'one_hot'
            - x_col(str): the name of the column that contains the observations (used only when the vectorizer is set to None)
                -- default: 'review'
            - y_col(str): the name of the column that contains the labels (used only when the vectorizer is set to None)
                --default: 'rating'
            - cut_off(int): the frequency-based filtering parameter (used only when the vectorizer is set to None) 
                -- default = 25
        '''
        
        self.review_df = review_df
        self._vectorizer = vectorizer
        if self._vectorizer is None:
            self._vectorizer = Vectorizer.from_dataframe(self.review_df, x_col, y_col, cut_off)
        
        self.x_col = x_col
        self.y_col = y_col 
        
        self.train_df = self.review_df[self.review_df['split']=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.review_df[self.review_df['split']=='val']
        self.val_size = len(self.val_df)
        
        self.test_df = self.review_df[self.review_df['split']=='test']
        self.test_size = len(self.test_df)
        
        self._lookup_dict = {
            'train': (self.train_df, self.train_size),
            'val': (self.val_df, self.val_size),
            'test': (self.test_df, self.test_size)
        }
        
        self.set_split('train')
        
    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv, **args):
        '''
        load the dataset from csv and create a vectorizer
        
        Args:
            - review_csv(str): dataset path
            - **args(dict): dictionary of arguments that will be passed to the class constructor
        
        Returns:
            - (ReviewDataset)
        '''
        
        review_df = pd.read_csv(review_csv)
        
        return cls(review_df, vectorizer=None, **args)

    @classmethod
    def load_dataset_and_load_vectorizer(cls, review_csv, vectorizer_path):
        '''
        load the dataset from csv and load its vectorizer
        
        Args:
            - review_csv(str): dataset path
            - vectorizer_path(str): the path to the vectorizer serializable
        
        Returns:
            - (ReviewDataset)
        '''
        
        review_df = pd.read_csv(review_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_path)
        
        return cls(review_df, vectorizer=vectorizer)
    
    @classmethod
    def load_vectorizer_only(cls, vectorizer_path):
        '''
        load the vectotizer from a json file
        
        Args:
            - vectorizer_path(str): the path to the file
        
        Returns:
            - (Vectorizer)
        '''
        with open(vectorizer_path, 'r') as f:
            return Vectorizer.from_serializable(json.load(f))
    
    def save_vectorizer(self, vectorizer_path):
        '''
        saves the vectorizer as a json file
        
        Args:
            vectorizer_path(str): the location to save the vectorizer
        '''
        with open(vectorizer_path, "w") as f:
            json.dump(self._vectorizer.to_serializable(), f)

    def get_vectorizer(self):
        ''' returns the vectorizer '''
        return self._vectorizer
    
    def set_split(self, split):
        '''
        select the dataset split - represented as a column in the dataframe -
        
        Args:
            - split(str)
        '''
        
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[self._target_split]
    
    def __len__(self):
        return self._target_size
    
    def __getitem__(self, idx):
        '''
        The primary entery point in pytorch dataset
        
        Args:
            - idx(int): the index of the data point
        
        Returns:
            - (dict): a dictionary that contains the data point's features (x_data) and labels (y_data)
        '''
        
        row = self._target_df.iloc[idx]
        
        review_vector = self._vectorizer.vectorize(row[self.x_col])
        rating_idx = self._vectorizer.y_vocab.lookup_token(row[self.y_col])
        
        return {
            'x_data': review_vector,
            'y_data': rating_idx
        }
    
    def get_num_batches(self, batch_size):
        '''
        given the batch size return the number of batches in the dataset
        
        Args:
            - batch_size(int)
        
        Returns:
            - num_batches(int)
        '''
        return len(self) // batch_size
     
def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device='cpu'):
    '''A generator function which wraps the PyTorch DataLoader'''

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)

        yield out_data_dict