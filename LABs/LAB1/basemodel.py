import numpy as np

class LinearModel:
    def __init__(self, in_features: int, out_features: int):
        """
        A linear base model.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        raise NotImplementedError("Subclasses must implement the initialization method.")

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the output given input.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def gradient(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> tuple:
        """
        Compute gradients for the loss.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True values/labels, shape [batch_size, out_features].
            predictions (np.ndarray): Predicted values, shape [batch_size, out_features].

        Returns:
            tuple: (dw, db), gradients for weights and bias.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement the gradient method.")

    def backpropagation(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray, learning_rate: float = 0.01) -> float:
        """
        Perform backpropagation, compute loss and update parameters.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True values/labels, shape [batch_size, out_features].
            learning_rate (float): Learning rate, default 0.01.

        Returns:
            float: Loss for the batch.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement the backpropagation method.")

    def save_checkpoint(self, filepath: str):
        """
        Save model parameters to file.

        Args:
            filepath (str): Path to save the checkpoint (.npz format).
        """
        np.savez(filepath, weight=self.weight, bias=self.bias)

    def load_checkpoint(self, filepath: str):
        """
        Load model parameters from file.

        Args:
            filepath (str): Path to the checkpoint file (.npz format).

        Raises:
            ValueError: If loaded parameters do not match model dimensions.
        """
        checkpoint = np.load(filepath)
        loaded_weight = checkpoint['weight']
        loaded_bias = checkpoint['bias']
        if loaded_weight.shape != self.weight.shape or loaded_bias.shape != self.bias.shape:
            raise ValueError("Loaded parameters do not match model dimensions.")
        self.weight = loaded_weight
        self.bias = loaded_bias





    