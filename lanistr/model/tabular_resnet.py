"""ResNet implementation for tabular data."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import dataclasses
import transformers

@dataclasses.dataclass
class ResNetOutput(transformers.utils.ModelOutput):
  """Base class for tabular resnet model outputs. To fit API."""
  last_hidden_state: Optional[torch.Tensor] = None

class ResidualBlock(nn.Module):
    """Residual block for tabular data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        """Initialize residual block.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
            activation: Activation function to use
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Activation {activation} not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Output tensor of shape [batch_size, input_dim]
        """
        # First linear layer with normalization and activation
        residual = x
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Second linear layer with normalization
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.dropout(x)

        # Add residual connection
        x = x + residual
        x = self.activation(x)
        
        return x


class ResNet(nn.Module):
    """ResNet for tabular data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        """Initialize ResNet.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension for residual blocks
            num_layers: Number of residual blocks
            dropout: Dropout rate
            activation: Activation function to use
        """
        super().__init__()
        
        # Initial projection layer to match dimensions if needed
        self.input_projection = None
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.input_dim = hidden_dim
        else:
            self.input_dim = input_dim

        # Stack of residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(
                input_dim=self.input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Output tensor of shape [batch_size, input_dim]
        """
        # Initial projection if needed
        if self.input_projection is not None:
            x = self.input_projection(x)

        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        return x


class TabularResNet(nn.Module):
    """TabularResNet with embedding support for categorical variables."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        cat_dims: list = None,
        cat_idxs: list = None,
        cat_emb_dim: int = None,
    ):
        """Initialize TabularResNet.

        Args:
            input_dim: Number of input features of data before categorical embedding
            hidden_dim: Hidden dimension for residual blocks
            output_dim: Desired output dimension
            num_layers: Number of residual blocks
            dropout: Dropout rate
            activation: Activation function to use
            cat_dims: List of dimensions for each categorical variable
            cat_idxs: List of indices for categorical variables
            cat_emb_dim: Embedding dimension for categorical variables
        """
        super().__init__()

        # Handle categorical variables
        self.cat_idxs = cat_idxs if cat_idxs is not None else []
        self.cat_dims = cat_dims if cat_dims is not None else []
        
        if len(self.cat_dims) != len(self.cat_idxs):
            raise ValueError("cat_dims and cat_idxs must have same length")

        # Create embeddings for categorical variables
        self.embeddings = nn.ModuleList()
        self.post_embed_dim = input_dim
        
        if self.cat_dims and self.cat_idxs:
            if isinstance(cat_emb_dim, int):
                self.cat_emb_dims = [cat_emb_dim] * len(self.cat_idxs)
            else:
                raise ValueError("cat_emb_dim must be an integer")

            for cat_dim, emb_dim in zip(self.cat_dims, self.cat_emb_dims):
                self.embeddings.append(nn.Embedding(cat_dim, emb_dim))
                # Adjust post embedding dimension
                self.post_embed_dim = self.post_embed_dim + emb_dim - 1

        # Create continuous feature mask
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[self.cat_idxs] = 0

        # Create ResNet
        self.resnet = ResNet(
            input_dim=self.post_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        if self.embeddings:
            # Handle categorical variables
            continuous_features = []
            categorical_features = []
            
            # Split continuous and categorical features
            for feat_idx, is_continuous in enumerate(self.continuous_idx):
                if is_continuous:
                    continuous_features.append(x[:, feat_idx].float().unsqueeze(1))
                else:
                    cat_idx = self.cat_idxs.index(feat_idx)
                    embedding = self.embeddings[cat_idx]
                    categorical_features.append(
                        embedding(x[:, feat_idx].long())
                    )

            # Concatenate all features
            x = torch.cat(continuous_features + categorical_features, dim=1)

        # Pass through ResNet
        return ResNetOutput(
            last_hidden_state=self.resnet(x),
        )
