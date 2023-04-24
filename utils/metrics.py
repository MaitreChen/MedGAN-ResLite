from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def classifier_evaluate(pred, label):
    accuracy = accuracy_score(label, pred, normalize=True)

    # The positive and negative sample ratio is balanced,so use macro
    average_mode = 'macro'
    precision = precision_score(label, pred, average=average_mode)
    recall = recall_score(label, pred, average=average_mode)
    f1 = f1_score(label, pred, average=average_mode)
    cm = confusion_matrix(label, pred, labels=None, sample_weight=None)

    return accuracy, precision, recall, f1, cm


def generator_evaluate():
    pass
