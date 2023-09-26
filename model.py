import numpy as np

class KMeans:
    def __init__(self, k: int, epsilon: float = 1e-6) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.epsilon = epsilon
    
    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:   # 100
        # Initialize cluster centers (need to be careful with the initialization,
        # otherwise you might see that none of the pixels are assigned to some
        # of the clusters, which will result in a division by zero error)
        self.cluster_centers = X[np.random.choice(len(X),self.num_clusters)]
        for _ in range(max_iter):
            # Assign each sample to the closest prototype

            centers = self.predict(X)
            
            # Update prototypes
            old_cc = self.cluster_centers
            new_cc = np.array([X[centers == i].mean(axis=0) for i in range(self.num_clusters)])
            diff = np.linalg.norm(new_cc-old_cc)
            self.cluster_centers = new_cc
            if(diff<self.epsilon):
                break

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predicts the index of the closest cluster center for each data point
        centers = np.empty((len(X)),dtype=int)
        d = np.empty((len(X),self.num_clusters))
        for i in range(self.num_clusters):
            d[:,i] = np.linalg.norm(X - self.cluster_centers[i,:], axis=1)
        centers = np.argmin(d,axis=1)
        return centers
        raise NotImplementedError
    
    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)
    
    def replace_with_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        # Returns an ndarray of the same shape as X
        # Each row of the output is the cluster center closest to the corresponding row in X
        centers = self.predict(X)
        for i in range(self.num_clusters):
            X[centers == i] = self.cluster_centers[i]
        return X
        raise NotImplementedError