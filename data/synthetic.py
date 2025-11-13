import torch
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, tasks, input_dim, correlation, num_samples, polynomial_degrees=None, noise_levels=None):
        """
        Initialize the synthetic dataset.
        Args:
            tasks (list): List of tasks (T).
            input_dim (int): Dimensionality of the input features (D).
            correlation (float): Desired pairwise cosine similarity (p).
            num_samples (int): Number of samples to generate.
            polynomial_degree (int): Degree of polynomial for polynomial-based labels.
            noise_levels (list): List of noise levels (one for each task).
        """
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.input_dim = input_dim
        self.correlation = correlation
        self.num_samples = num_samples
        self.polynomial_degrees = polynomial_degrees if polynomial_degrees else [2] * self.num_tasks
        self.noise_levels = noise_levels if noise_levels else [0.01] * self.num_tasks

        # Generate orthogonal unit vectors U
        self.U = self._generate_orthogonal_unit_vectors()

        # Construct the Gram matrix P
        self.P = self._construct_gram_matrix()

        # Perform eigendecomposition to construct weight matrix W
        self.W = self._construct_weight_matrix()

        # Generate input features
        self.X = torch.randn(self.num_samples, self.input_dim)

        # Generate labels
        self.Y = self._generate_labels()

    def _generate_orthogonal_unit_vectors(self):
        """Generate orthogonal unit vectors U."""
        U = torch.randn(self.num_tasks, self.input_dim)
        for i in range(self.num_tasks):
            for j in range(i):
                U[i] -= torch.dot(U[i], U[j]) * U[j]  # Orthogonalize against previous vectors
            U[i] /= U[i].norm()  # Normalize to unit length
        return U

    def _construct_gram_matrix(self):
        """Construct the Gram matrix P."""
        P = torch.full((self.num_tasks, self.num_tasks), self.correlation)
        P.fill_diagonal_(1.0)
        return P

    def _construct_weight_matrix(self):
        """Construct the weight matrix W."""
        eigvals, eigvecs = torch.linalg.eigh(self.P)  # Eigendecomposition of P
        sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=0.0))  # Eigenvalues are guaranteed to be non-negative
        W = eigvecs @ torch.diag(sqrt_eigvals) @ self.U  # Combine eigencomponents with U
        return W

    def _generate_labels(self):
        """Generate labels for each task."""
        Y = []
        for i in range(self.num_tasks):
            # Feature-wise polynomial-based labels
            linear_term = self.X @ self.W[i]
            poly_term = sum((linear_term ** k) for k in range(2, self.polynomial_degrees[i] + 1))
            noise = torch.randn(self.num_samples) * self.noise_levels[i]
            Y.append(linear_term + poly_term + noise)

        return torch.stack(Y, dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {'features': self.X[idx], 'targets': {task: self.Y[idx, i] for i, task in enumerate(self.tasks)}}