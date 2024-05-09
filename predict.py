import torch
from preprocess_data import preprocess_text
def predict(classifier, review, vectorizer, decision_threshold, args):
    '''predict the rating of the review'''
    
    review = preprocess_text(review)
    vectorized_review = torch.tensor(vectorizer.vectorize(review), dtype=torch.float32).to(args.device)

    pred = classifier(vectorized_review.view(1,-1))
    pred = torch.sigmoid(pred).item()
    index = 1 if pred >= decision_threshold else 0
    
    return vectorizer.y_vocab.lookup_index(index) 