# model.py
"""
Contains the FaultDetector neural network definition.
All architecture changes should be made here.
"""

import torch.nn as nn

import config


class FaultDetector(nn.Module):
    """Binary classifier for electrical fault detection.

    A compact feed-forward network that maps engineered electrical features
    to a single raw logit.  Use ``BCEWithLogitsLoss`` during training —
    the sigmoid activation is intentionally excluded from the forward pass.

    Parameters
    ----------
    input_dim : int
        Number of input features (determined at runtime from the experiment
        feature list).
    dropout : float, optional
        Dropout probability. Defaults to ``config.DROPOUT``. Several
        experiments use only 3-6 input features, where a high dropout rate
        removes an entire feature from a large share of forward passes and
        destabilises training.

    Examples
    --------
    >>> model = FaultDetector(input_dim=8)
    >>> logits = model(x_batch)   # shape: (batch_size,)
    """

    def __init__(self, input_dim: int, dropout: float | None = None) -> None:
        super().__init__()
        p = config.DROPOUT if dropout is None else dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p),

            # Optional second hidden layer — uncomment to enable:
            # nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Dropout(p),

            nn.Linear(64, 1),  # raw logit; BCEWithLogitsLoss handles sigmoid
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size,)`` — squeezed raw logits.
        """
        return self.net(x).squeeze(1)
