import torch
import torch.nn as nn


class ReviewClassifier(nn.Module):
    def __init__(self, num_features):
        '''
        Args:
            - num_features(int)
        '''
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x_in, apply_sigmoid=False):
        '''
        The forward pass of the classifier
        
        Args:
            - x_in (torch.Tensor): an input data tensor. 
                -- x_in.shape should be (batch, num_features)
            - apply_sigmoid (bool): a flag for the sigmoid activation
                -- should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch,)
        '''
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out
    
    def compute_accuracy(self, y_target, y_pred):
        ''' given targets and predictions(probabilities) it computes the accuracy'''
        y_target = y_target.cpu()
        y_pred_indices = torch.sigmoid(y_pred >= 0.5).cpu()
        n_correct = torch.eq(y_target, y_pred_indices).sum().item()
        return (n_correct / len(y_pred_indices)) * 100