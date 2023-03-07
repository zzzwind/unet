import torch

def _confusion_matrix(pred, label, class_id):
    TP = ((pred == class_id) & (label == class_id)).sum().item()
    FP = ((pred == class_id) & (label != class_id)).sum().item()
    TN = ((pred != class_id) & (label != class_id)).sum().item()
    FN = ((pred != class_id) & (label == class_id)).sum().item()
    return TP, FP, TN, FN

def get_confusion_matrix(pred, label, num_classes):
    pred = torch.argmax(pred, dim=1)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int)
    for i in range(num_classes):
        TP, FP, TN, FN = _confusion_matrix(pred, label, i)
        cm[i][i] = TP
        cm[i][0:i] = FN / (num_classes - 1)
        cm[i][i + 1:] = FP / (num_classes - 1)
    return cm

def get_pixel_accuracy(pred, label, num_classes):
    confusion_matrix = get_confusion_matrix(pred, label, num_classes)
    return torch.diag(confusion_matrix).sum() / confusion_matrix.sum()


def recall(pred, label, num_classes):
    # confusion_matrix: shape=(num_classes, num_classes)
    # return: recall
    confusion_matrix = get_confusion_matrix(pred, label, num_classes)
    recall = torch.diag(confusion_matrix) / (confusion_matrix.sum(dim=1) + 1e-20)
    return torch.mean(recall)

def f1_score(pred, label, num_classes):
    # confusion_matrix: shape=(num_classes, num_classes)
    # return: F1-score
    confusion_matrix = get_confusion_matrix(pred, label, num_classes)
    precision = torch.diag(confusion_matrix) / (confusion_matrix.sum(dim=0) + 1e-20)
    recall = torch.diag(confusion_matrix) / (confusion_matrix.sum(dim=1) + 1e-20)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-20)
    return torch.mean(f1)

def miou(pred, label, num_classes):
    # confusion_matrix: shape=(num_classes, num_classes)
    # return: mIoU
    confusion_matrix = get_confusion_matrix(pred, label, num_classes)
    intersection = torch.diag(confusion_matrix)
    union = confusion_matrix.sum(dim=0) + confusion_matrix.sum(dim=1) - intersection
    iou = intersection / (union + 1e-20)
    return iou.mean()

if __name__ == "__main__":
    pred = torch.randn(2, 2, 256, 256)
    label = torch.randn(2, 256, 256)
    cm = get_confusion_matrix(pred, label, 2)
    pa = get_pixel_accuracy(pred=pred, label=label, num_classes=2)
    rc = recall(pred=pred,label=label, num_classes=2)
    f1 = f1_score(pred,label,2)
    mi = miou(pred,label,2)
    recall = recall(cm)
    miou = miou(cm)
    print(cm)
