from sklearn.metrics import f1_score

def evaluate_f1(preds, labels):
    return f1_score(labels, preds, average='weighted')
