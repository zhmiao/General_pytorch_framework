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


def get_algorithm(name, args):

    """
    Algorithm getter
    """

    alg = algorithms[name](args)
    return alg


class Algorithm:

    """
    Base Algorithm class for reference.
    """

    name = None

    def __init__(self, args):
        self.args = args
        self.logger = self.args.logger
        self.weights_path = './weights/{}/{}_{}.pth'.format(self.args.algorithm, self.args.conf_id, self.args.session)

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def train_epoch(self, epoch):
        pass

    def train(self):
        pass

    def evaluate_epoch(self, loader):
        pass

    def evaluate(self, loader):
        pass

    def save_model(self):
        pass


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
