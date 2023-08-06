import torchmetrics
import torch



device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_metrics(config, *args, **kwargs):
    metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.F1Score(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            torchmetrics.Accuracy(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            torchmetrics.Dice(),
            torchmetrics.Precision(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            torchmetrics.Specificity(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            torchmetrics.Recall(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            # IoU
            torchmetrics.JaccardIndex(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
        ],
        prefix="metrics/",
    )

    # test_metrics
    test_metrics = metrics.clone(prefix="").to(device)


def get_binary_metrics(*args, **kwargs):
    metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.F1Score(task="binary"),
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.Dice(multiclass=False),
            torchmetrics.Precision(task="binary"),
            torchmetrics.Specificity(task="binary"),
            torchmetrics.Recall(task="binary"),
            # IoU
            torchmetrics.JaccardIndex(task="binary", num_labels=2, num_classes=2),
        ],
        prefix="metrics/",
    )

    # test_metrics
    test_metrics = metrics.clone(prefix="").to(device)
    return test_metrics
