import numpy as np

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None, None

        parent_mse = np.var(y)

        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                weighted_mse = (len(y_left) / n_samples) * np.var(y_left) + (len(y_right) / n_samples) * np.var(y_right)
                gain = parent_mse - weighted_mse

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return {'value': np.mean(y)}

        split_idx, split_thresh = self._best_split(X, y)
        if split_idx is None:
            return {'value': np.mean(y)}

        left_mask = X[:, split_idx] <= split_thresh
        right_mask = ~left_mask

        return {
            'feature': split_idx,
            'threshold': split_thresh,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _predict_row(self, tree, x):
        if 'value' in tree:
            return tree['value']
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_row(tree['left'], x)
        return self._predict_row(tree['right'], x)

    def predict(self, X):
        return np.array([self._predict_row(self.tree, x) for x in X])


class XGBoostFromScratch:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_pred = None

    def _sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))