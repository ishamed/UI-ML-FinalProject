import numpy as np

class XGBoostNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, val=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.val = val  # وزن برگ (Leaf Weight)

    def is_leaf(self):
        return self.val is not None

class XGBoostTree:
    def __init__(self, max_depth=3, lambda_reg=1.0, gamma=0.0, min_child_weight=1.0):
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.root = None

    def _compute_leaf_weight(self, g, h):
        return -np.sum(g) / (np.sum(h) + self.lambda_reg)
    