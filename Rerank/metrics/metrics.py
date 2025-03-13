from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from rouge import Rouge


def binary_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def multi_class_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro")
    }

def ordinal_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False)
    }

def rouge_scores(hypotheses, references):
    rouge = Rouge()
    scores = rouge.get_scores(hypotheses, references, avg=True)
    return {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-l": scores["rouge-l"]["f"]
    }