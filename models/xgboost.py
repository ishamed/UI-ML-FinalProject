import numpy as np

class XGBoostNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, val=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.val = val

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

    def _similarity_score(self, g, h):
        return (np.sum(g) ** 2) / (np.sum(h) + self.lambda_reg)

    def fit(self, X, g, h):
        self.root = self._grow_tree(X, g, h, depth=0)

    def _grow_tree(self, X, g, h, depth):
        if depth >= self.max_depth or np.sum(h) < self.min_child_weight or len(np.unique(g)) == 1:
            return XGBoostNode(val=self._compute_leaf_weight(g, h))

        best_feat, best_thresh, best_gain = None, None, -float('inf')
        root_similarity = self._similarity_score(g, h)
        n_features = X.shape[1]

        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            for thr in thresholds:
                left_mask = X[:, feat_idx] <= thr
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                g_l, h_l = g[left_mask], h[left_mask]
                g_r, h_r = g[right_mask], h[right_mask]

                if np.sum(h_l) < self.min_child_weight or np.sum(h_r) < self.min_child_weight:
                    continue


                gain = 0.5 * (self._similarity_score(g_l, h_l) + self._similarity_score(g_r, h_r) - root_similarity) - self.gamma
                if gain > best_gain and gain > 0:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thr

        if best_feat is None:
            return XGBoostNode(val=self._compute_leaf_weight(g, h))

        left_mask = X[:, best_feat] <= best_thresh
        left = self._grow_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        right = self._grow_tree(X[~left_mask], g[~left_mask], h[~left_mask], depth + 1)
        return XGBoostNode(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def predict(self, X):
        return np.array([self._predict_row(self.root, x) for x in X])

    def _predict_row(self, node, x):
        if node.is_leaf():
            return node.val
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)


class XGBoostClassifier:
    def __init__(self, n_estimators=15, learning_rate=0.1, max_depth=3, lambda_reg=1.0, gamma=0.0):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.trees = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        y_pred_log_odds = np.zeros(len(y))

        for _ in range(self.n_estimators):
            p = self._sigmoid(y_pred_log_odds)


            g = p - y
            h = p * (1 - p) + 1e-16

            tree = XGBoostTree(
                max_depth=self.max_depth,
                lambda_reg=self.lambda_reg,
                gamma=self.gamma
            )
            tree.fit(X, g, h)

            update = tree.predict(X)
            y_pred_log_odds += self.lr * update
            self.trees.append(tree)

    def predict_proba(self, X):
        X = np.array(X)
        y_pred_log_odds = np.zeros(len(X))
        for tree in self.trees:
            y_pred_log_odds += self.lr * tree.predict(X)
        return self._sigmoid(y_pred_log_odds)

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return np.where(probas >= threshold, 1, 0)
