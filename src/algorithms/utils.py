import numpy as np


algorithms = {}


def register_algorithm(name):

    """
    Algorithm register
    """

    def decorator(cls):
        algorithms[name] = cls
        return cls
    return decorator


def get_algorithm(name, **kwargs):

    """
    Algorithm getter
    """

    alg = algorithms[name](**kwargs)
    return alg


def acc(preds, labels, num_classes=6):

    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # class_correct = np.array([0. for _ in range(len(label_counts))])
    class_correct = np.array([0. for _ in range(num_classes)])

    for p, l in zip(preds, labels):
        if p == l:
            class_correct[l] += 1

    class_acc = class_correct[unique_labels] / label_counts

    mac_acc = class_acc.mean()
    mic_acc = class_correct.sum() / label_counts.sum()

    return class_acc, mac_acc, mic_acc, unique_labels