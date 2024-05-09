import torch
import os
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from utils.utils import *
from model.dataset_loader import ReviewDataset

class TrainState:
    ''' this class is responsible for defining and updating train state'''
    
    def __init__(self, args):
        self.args = args
        self._state = {
            'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': self.args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': self.args.model_state_file
        }
    
    def update(self, model):
        ''' handles the train state updates (early stopping - model checkpoints) '''
        
        # Saving initial model
        if self._state['epoch_index'] == 0:
            self._save_model(model)
        
        # Handling early stopping
        if self._state['epoch_index'] > 0:
            val_loss = self._state['val_loss'][-1]
            
            # if loss is worst
            if val_loss >= self._state['early_stopping_best_val']:
                self._state['early_stopping_step'] += 1
            
            # if loss is better
            else:
                self._save_model(model)
                self._state['early_stopping_step'] = 0
        
        self._state['stop_early'] = self._state['early_stopping_step'] == self.args.early_stopping_criteria
        
        return self
    
    def _save_model(self, model):
        torch.save(model.state_dict(), self._state['model_filename'])
    
    def is_early_stopping(self):
        return self._state['stop_early']
    
    def __getitem__(self, item):
        return self._state[item]
    
    def __setitem__(self, item, val):
        self._state[item] = val


def expand_filepaths_to_save_dir(args):
    args.vectorizer_file = os.path.join(args.save_dir,
                                        args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)
    
    print("Expanded filepaths: ")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))
    
    return args

def prepare_cuda(args):
    if not torch.cuda.is_available():
        args.cuda = False

    print("Using CUDA: {}".format(args.cuda))

    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    return args

def setup_training(args):
    expand_filepaths_to_save_dir(args)
    prepare_cuda(args)
    set_seed_everywhere(args.seed, args.cuda)
    handle_dirs(args.save_dir)
    
    return args


def get_training_progress_bars(args, dataset):
    
    tqdm_func = tqdm_notebook if in_jupyter_notebook() else tqdm
    
    epoch_bar = tqdm_func(desc='training routine', 
                          total=args.num_epochs,
                          position=0)

    dataset.set_split('train')
    train_bar = tqdm_func(desc='split=train',
                              total=dataset.get_num_batches(args.batch_size), 
                              position=1, 
                              leave=True)
    dataset.set_split('val')
    val_bar = tqdm_func(desc='split=val',
                            total=dataset.get_num_batches(args.batch_size), 
                            position=1, 
                            leave=True)
    
    return epoch_bar, train_bar, val_bar


def load_data_and_vectorizer(args):
    if args.reload_from_files:
        # training from a checkpoint
        print("Loading dataset and vectorizer")
        dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv,
                                                                args.vectorizer_file)
    else:
        # create dataset and vectorizer
        print("Loading dataset and creating vectorizer")
        dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
        dataset.save_vectorizer(args.vectorizer_file)  
        
    vectorizer = dataset.get_vectorizer()
    
    return dataset, vectorizer