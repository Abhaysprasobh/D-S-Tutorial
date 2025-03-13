import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def compute_lda_manual(X, y):
    # Separate classes
    class_labels = np.unique(y)
    mean_vectors = []
    for label in class_labels:
        mean_vectors.append(np.mean(X[y == label], axis=0))
    
    # Compute S_W (Within-class scatter matrix)
    S_W = np.zeros((X.shape[1], X.shape[1]))
    for label, mean_vec in zip(class_labels, mean_vectors):
        class_scatter = np.cov(X[y == label].T, bias=True) * (X[y == label].shape[0] - 1)
        S_W += class_scatter
    
    # Compute S_B (Between-class scatter matrix)
    overall_mean = np.mean(X, axis=0)
    S_B = np.zeros((X.shape[1], X.shape[1]))
    for label, mean_vec in zip(class_labels, mean_vectors):
        n = X[y == label].shape[0]
        mean_diff = (mean_vec - overall_mean).reshape(-1, 1)
        S_B += n * (mean_diff @ mean_diff.T)
    
    # Solve for eigenvalues and eigenvectors using pseudo-inverse
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_W) @ S_B)
    
    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(-eigvals)
    lda_direction = eigvecs[:, sorted_indices[0]]  # First eigenvector
    
    return lda_direction

# Example 2x2 dataset
X = np.array([[4, 2], [2, 4], [6, 2], [2, 6]])
y = np.array([0, 0, 1, 1])

# Compute LDA manually
lda_manual = compute_lda_manual(X, y)
print("LDA Direction (Manual Calculation):", lda_manual)

# Compute LDA using Scikit-learn
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X, y)
print("LDA Direction (Scikit-learn):", lda.scalings_.flatten())
