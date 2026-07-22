import numpy as np

class DecisionTreeLeaf:
    def __init__(self, value):
        self.value = value


class DecisionTreeNode:
    def __init__(self, feature_idx, threshold, left, right):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right


class DecisionTreeForForest:
    def __init__(self, max_depth=7, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        p1 = np.mean(y == 1)
        p0 = 1 - p1
        return 1.0 - (p1 ** 2 + p0 ** 2)

    def _best_split(self, X, y):
        best_gain = -1.0
        split_idx, split_thresh = None, None
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None, None

        parent_gini = self._gini_impurity(y)

        if self.max_features is None:
            feat_indices = np.arange(n_features)
        else:
            feat_indices = np.random.choice(n_features, self.max_features, replace=False)

        for feat_idx in feat_indices:
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                weighted_gini = (len(y_left) / n_samples) * self._gini_impurity(y_left) + \
                                (len(y_right) / n_samples) * self._gini_impurity(y_right)

                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            leaf_value = 1 if np.mean(y) >= 0.5 else 0
            return DecisionTreeLeaf(leaf_value)

        split_idx, split_thresh = self._best_split(X, y)
        if split_idx is None:
            leaf_value = 1 if np.mean(y) >= 0.5 else 0
            return DecisionTreeLeaf(leaf_value)

        left_mask = X[:, split_idx] <= split_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(split_idx, split_thresh, left_child, right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_row(self, node, x):
        if isinstance(node, DecisionTreeLeaf):
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X):
        return np.array([self._predict_row(self.root, x) for x in X])