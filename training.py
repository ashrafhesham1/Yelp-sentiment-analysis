import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from argparse import Namespace
from model.dataset_loader import generate_batches
from model.model import ReviewClassifier
from utils.utils import *
from utils.train_utils import *

if __name__=='__main__':
    args = Namespace(
    # Data and Path information
    frequency_cutoff=25,
    model_state_file='model.pth',
    review_csv='yelp/reviews_with_splits_lite.csv',
    save_dir='model_storage/ch3/yelp/',
    vectorizer_file='vectorizer.json',
    # No Model hyper parameters
    # Training hyper parameters
    batch_size=128,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=100,
    seed=1337,
    # Runtime options
    catch_keyboard_interrupt=True,
    cuda=True,
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
    )
    
    setup_training(args)
    dataset, vectorizer = load_data_and_vectorizer(args)

    classifier = ReviewClassifier(num_features=len(vectorizer.x_vocab))
    classifier = classifier.to(args.device)

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    mode='min', factor=0.5,
                                                    patience=1)
    
    # training loop
    train_State = TrainState(args)
    epoch_bar, train_bar, val_bar = get_training_progress_bars(args, dataset)

    try:
        for epoch_index in range(args.num_epochs):
            train_State['epoch_index'] = epoch_index
            
            dataset.set_split('train')
            batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
            
            running_loss, running_acc = 0.0, 0.0
            classifier.train()
            
            for batch_index, batch_dict in enumerate(batch_generator):
                optimizer.zero_grad()
                
                y_pred = classifier(x_in=batch_dict['x_data'].float())
                loss = loss_func(y_pred, batch_dict['y_data'].float())
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)
                
                loss.backward()
                optimizer.step()
                
                acc_t = classifier.compute_accuracy(batch_dict['y_data'], y_pred)
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                
                train_bar.set_postfix(loss=running_loss, acc=running_acc, epoch = epoch_index)
                train_bar.update()
                
            train_State['train_loss'].append(running_loss)
            train_State['train_acc'].append(running_acc)
                
            # Compute validation loss    
            dataset.set_split('val')
            batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
            
            running_loss, running_acc = 0.0, 0.0
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                
                y_pred = classifier(x_in=batch_dict['x_data'].float())
                loss = loss_func(y_pred, batch_dict['y_data'].float())
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)
                
                acc_t = classifier.compute_accuracy(batch_dict['y_data'], y_pred)
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                
                val_bar.set_postfix(loss=running_loss, acc=running_acc, epoch = epoch_index)
                val_bar.update()
                
            train_State['val_loss'].append(running_loss)
            train_State['val_acc'].append(running_acc)
            
            # update train state and check early stopping
            train_State.update(classifier)
            scheduler.step(train_State['val_loss'][-1])
            
            train_bar.n, val_bar.n = 0, 0
            epoch_bar.update()
            
            if train_State.is_early_stopping():
                break
            
    except KeyboardInterrupt:
        print('Exiting loop')