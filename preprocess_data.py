import collections
import numpy as np
import pandas as pd
import re
from argparse import Namespace

args = Namespace(
    raw_train_dataset_csv="yelp/raw_train.csv",
    raw_test_dataset_csv="yelp/raw_test.csv",
    proportion_subset_of_train=0.1,
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="yelp/reviews_with_splits_lite.csv",
    seed=1337
)


np.random.seed(args.seed)


###################
    # Functions
###################

def get_reviews_by_rating(train_reviews):
    '''given reviews dataframe it returns a dictionary that maps each rating to a list of reviews with this rating'''
    by_rating = collections.defaultdict(list)
    for _, row in train_reviews.iterrows():
        by_rating[row.rating].append(row.to_dict())
    
    return by_rating


def get_data_subset(train_reviews):
    ''' gets a subset of the data with proportion defined in args'''
    by_rating = get_reviews_by_rating(train_reviews)
    
    review_subset = []
    for _, item_list in sorted(by_rating.items()):
        n_total = len(item_list)
        n_subset = int(n_total * args.proportion_subset_of_train)
        review_subset.extend(item_list[:n_subset])
    
    return pd.DataFrame(review_subset)


def create_splits(train_reviews):
    '''given a dataframe of reviews it splits it into traint/val/test '''
    by_rating = get_reviews_by_rating(train_reviews)
    
    final_list = []
    for _, item_list in sorted(by_rating.items()):
        
        np.random.shuffle(item_list)
        
        n_total = len(item_list)
        n_train = int(n_total * args.train_proportion)
        n_val = int(n_total * args.val_proportion)
        n_test = int(n_total * args.test_proportion)
        
        # assign each data point to a split attribute
        for item in item_list[:n_train]:
            item['split'] = 'train'

        for item in item_list[n_train:n_train+n_val]:
            item['split'] = 'val'

        for item in item_list[n_train+n_val:n_train+n_val+n_test]:
            item['split'] = 'test'
        
        final_list.extend(item_list)
    
    return pd.DataFrame(final_list)


def preprocess_text(text):
    '''perform some cleaning on the reviews '''
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


def pipeline(train_reviews):
    ''' apply all the preprocessing steps to the reviews dataframe'''
    reviews_subset = get_data_subset(train_reviews)
    reviews_with_splits = create_splits(reviews_subset)
    
    reviews_with_splits['review'] = reviews_with_splits['review'].apply(preprocess_text)
    reviews_with_splits['rating'] = reviews_with_splits['rating'].apply({1: 'negative', 2: 'positive'}.get)

    return reviews_with_splits



if __name__=='__main__':
    print(f'# Starting preprocessing: {args.raw_train_dataset_csv}......')
    train_reviews = pd.read_csv(args.raw_train_dataset_csv, header=None, names=['rating', 'review'])
    reviews_with_splits = pipeline(train_reviews)
    reviews_with_splits.to_csv(args.output_munged_csv, index=False)
    print(f'# Done - preprocessed data: {args.output_munged_csv}')