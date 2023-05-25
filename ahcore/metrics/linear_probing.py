"""implementaion of linear probing as metric and loss"""

import torch
from sklearn import linear_model, metrics

# from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn


class LinearProbing(nn.Module):
    """Module used for linear probing during training"""

    def __init__(self, model_name, test_size: float = 0.3, scoring="accuracy", **kwargs) -> None:
        """
        Params:
            model_name: name of the linear model to be used
            test_size: portion of data to use as test data
            scoring: string specifying the scoring to be performed (e.g. accuracy, auc),
                default accuracy
            kwargs: kwargs to be passed to the linear model
        """
        super().__init__()
        self.test_size = test_size
        self.clf = getattr(linear_model, model_name)(**kwargs)
        self.scorer = metrics.get_scorer(scoring)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """runs the linear probing"""
        input_np, target_np = input.detach().cpu().numpy(), target.detach().cpu().numpy()
        # ensure stratified splitting of target classes
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size)
        indices_train, indices_test = next(splitter.split(input_np, target_np))
        x_train, y_train = input_np[indices_train], target_np[indices_train]
        x_test, y_test = input_np[indices_test], target_np[indices_test]
        try:
            self.clf.fit(x_train, y_train)
        except ValueError:  # data may only contain one class
            return torch.tensor(0.0)
        return torch.tensor(self.scorer(self.clf, x_test, y_test))
