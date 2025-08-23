import numpy as np
from datasets import load_metric
from evaluate import load as load_evaluate_metric

def compute_metrics_classification(eval_pred):
    metric_acc = load_metric("accuracy")
    metric_prec = load_metric("precision")
    metric_rec = load_metric("recall")
    metric_f1 = load_metric("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = metric_prec.compute(predictions=predictions, references=labels, average="macro")["precision"]
    recall = metric_rec.compute(predictions=predictions, references=labels, average="macro")["recall"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    accuracy = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

def compute_metrics_token_classification(eval_pred, label_list):
    seqeval = load_evaluate_metric("seqeval")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }