"""
Transfer Learning Implementation with Progressive Neural Networks
Enables knowledge transfer across different game domains.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import copy


class ProgressiveNeuralNetwork(nn.Module):
    """
    Progressive Neural Network for transfer learning.

    Architecture:
    - Task-specific columns (one per domain)
    - Each column has its own parameters
    - Lateral connections allow knowledge transfer from previous columns
    - Can add new columns for new domains without forgetting old knowledge
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        num_tasks: int = 2,
        use_lateral: bool = True,
    ):
        """
        Initialize progressive neural network.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension (action space)
            hidden_dims: Hidden layer dimensions for each task column
            num_tasks: Number of tasks/domains
            use_lateral: Whether to use lateral connections
        """
        super(ProgressiveNeuralNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_tasks = num_tasks
        self.use_lateral = use_lateral

        self.num_layers = len(hidden_dims) + 1

        # Task-specific columns
        self.task_columns = nn.ModuleList()

        # Lateral connections (from previous columns)
        self.lateral_connections = nn.ModuleList()

        for task_id in range(num_tasks):
            # Build column for this task
            column = nn.ModuleList()

            prev_dim = input_dim
            for layer_idx, hidden_dim in enumerate(hidden_dims):
                column.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim

            # Output layer
            column.append(nn.Linear(prev_dim, output_dim))
            self.task_columns.append(column)

            # Lateral connections from previous tasks
            if use_lateral and task_id > 0:
                laterals = nn.ModuleList()

                for layer_idx in range(self.num_layers - 1):  # Exclude output layer
                    hidden_dim = hidden_dims[layer_idx]
                    # Lateral: takes output of previous task's layer and projects to current hidden_dim
                    lateral = nn.Linear(hidden_dims[layer_idx], hidden_dim)
                    laterals.append(lateral)

                self.lateral_connections.append(laterals)
            else:
                self.lateral_connections.append(nn.ModuleList())

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        """
        Forward pass through progressive network.

        Args:
            x: Input tensor
            task_id: Which task column to use

        Returns:
            Output logits
        """
        # Get task column
        column = self.task_columns[task_id]

        # Forward through layers with lateral connections
        activations = []
        activation = x

        for layer_idx in range(self.num_layers - 1):  # Exclude output layer
            # Task-specific layer
            activation = column[layer_idx](activation)

            # Add lateral input from previous tasks
            if self.use_lateral and task_id > 0:
                lateral_input = activations[layer_idx]  # Output of previous task's layer
                lateral_output = self.lateral_connections[task_id - 1][layer_idx](lateral_input)
                activation = activation + lateral_output  # Residual connection

            activation = torch.relu(activation)
            activations.append(activation)

        # Output layer
        output = column[-1](activations[-1])

        return output

    def forward_all_tasks(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward through all task columns"""
        outputs = []
        for task_id in range(self.num_tasks):
            output = self.forward(x, task_id)
            outputs.append(output)
        return outputs


class TransferLearner:
    """
    Manages transfer learning from source to target domain.

    Strategies:
    1. Fine-tuning: Train only top layers
    2. Feature extraction: Freeze base, train top only
    3. Progressive networks: Add new layers for new domain
    """

    def __init__(
        self,
        source_model: nn.Module,
        target_model: nn.Module,
        transfer_strategy: str = 'fine_tuning',  # 'fine_tuning', 'feature_extraction', 'progressive'
        learning_rate: float = 1e-4,
        device: str = 'cpu',
    ):
        """
        Initialize transfer learner.

        Args:
            source_model: Pre-trained source domain model
            target_model: Model for target domain
            transfer_strategy: Transfer learning strategy
            learning_rate: Learning rate for target domain training
            device: Torch device
        """
        self.source_model = source_model
        self.target_model = target_model
        self.transfer_strategy = transfer_strategy
        self.device = device

        # Copy source weights to target (partial if not all layers match)
        self._initialize_target_from_source()

        # Setup optimizer
        if transfer_strategy == 'fine_tuning':
            # All parameters are trainable
            self.optimizer = torch.optim.Adam(
                target_model.parameters(),
                lr=learning_rate
            )
            self.trainable_params = list(target_model.parameters())

        elif transfer_strategy == 'feature_extraction':
            # Only train top layers (typically last 2 layers)
            self.trainable_params = []
            for name, param in target_model.named_parameters():
                if 'layer' in name and int(name.split('layer')[1][0]) >= 2:
                    param.requires_grad = True
                    self.trainable_params.append(param)
                else:
                    param.requires_grad = False

            self.optimizer = torch.optim.Adam(
                self.trainable_params,
                lr=learning_rate * 10  # Higher LR for top layers
            )

        self.training_stats = {
            'source_performance': [],
            'target_performance': [],
            'transfer_loss': [],
        }

    def _initialize_target_from_source(self):
        """Copy weights from source model to target where possible"""
        source_state = self.source_model.state_dict()
        target_state = self.target_model.state_dict()

        # Copy matching parameters
        copied_count = 0
        for name in target_state:
            if name in source_state:
                target_state[name] = source_state[name]
                copied_count += 1

        self.target_model.load_state_dict(target_state)
        print(f"Transferred {copied_count} parameters from source to target model")

    def compute_transfer_loss(
        self,
        source_output: torch.Tensor,
        target_output: torch.Tensor,
        method: str = 'mmd',  # 'mmd', 'coral', 'wasserstein'
    ) -> torch.Tensor:
        """
        Compute domain transfer loss (discrepancy between source and target distributions).

        Args:
            source_output: Source model output [batch_size, dim]
            target_output: Target model output [batch_size, dim]
            method: Transfer loss method

        Returns:
            Transfer loss value
        """
        if method == 'mmd':
            # Maximum Mean Discrepancy
            return self._mmd_loss(source_output, target_output)

        elif method == 'coral':
            # CORAL loss (Correlation Alignment)
            return self._coral_loss(source_output, target_output)

        elif method == 'wasserstein':
            # Wasserstein distance (simplified)
            return torch.abs(source_output.mean() - target_output.mean())

        else:
            return torch.tensor(0.0)

    def _mmd_loss(self, source: torch.Tensor, target: torch.Tensor, kernel: str = 'rbf') -> torch.Tensor:
        """Maximum Mean Discrepancy loss"""
        batch_size = source.shape[0]

        if kernel == 'rbf':
            gamma = 1.0 / source.shape[-1]

            # Compute RBF kernel
            xx = torch.mm(source, source.t())
            yy = torch.mm(target, target.t())
            xy = torch.mm(source, target.t())

            rx = (xx.diag().unsqueeze(0) - 2 * xx + xx.diag().unsqueeze(1)) * gamma
            ry = (yy.diag().unsqueeze(0) - 2 * yy + yy.diag().unsqueeze(1)) * gamma
            rxy = (xx.diag().unsqueeze(0) - 2 * xy + yy.diag().unsqueeze(1)) * gamma

            k_xx = torch.exp(-rx.clamp(min=0))
            k_yy = torch.exp(-ry.clamp(min=0))
            k_xy = torch.exp(-rxy.clamp(min=0))

        mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

        return mmd

    def _coral_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """CORAL loss (Correlation Alignment)"""
        # Compute covariance matrices
        source_mean = source.mean(0, keepdim=True)
        target_mean = target.mean(0, keepdim=True)

        source_centered = source - source_mean
        target_centered = target - target_mean

        source_cov = torch.mm(source_centered.t(), source_centered)
        target_cov = torch.mm(target_centered.t(), target_centered)

        # CORAL loss: Frobenius norm of difference
        coral_loss = torch.norm(source_cov - target_cov, p='fro')

        return coral_loss

    def update(
        self,
        source_inputs: torch.Tensor,
        target_inputs: torch.Tensor,
        target_labels: torch.Tensor,
        loss_fn: Any,
        transfer_weight: float = 0.1,
    ) -> Dict[str, float]:
        """
        Perform one update step with transfer learning.

        Args:
            source_inputs: Source domain inputs
            target_inputs: Target domain inputs
            target_labels: Target domain labels
            loss_fn: Classification/RL loss function
            transfer_weight: Weight of transfer loss

        Returns:
            Loss dictionary
        """
        # Forward pass
        source_output = self.source_model(source_inputs.to(self.device))
        target_output = self.target_model(target_inputs.to(self.device))

        # Task loss (standard classification/RL loss)
        task_loss = loss_fn(target_output, target_labels.to(self.device))

        # Transfer loss
        transfer_loss = self.compute_transfer_loss(source_output, target_output)

        # Combined loss
        total_loss = task_loss + transfer_weight * transfer_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
        self.optimizer.step()

        return {
            'task_loss': task_loss.item(),
            'transfer_loss': transfer_loss.item(),
            'total_loss': total_loss.item(),
        }

    def evaluate(
        self,
        source_inputs: torch.Tensor,
        source_labels: torch.Tensor,
        target_inputs: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate transfer learning effectiveness.

        Returns:
            Performance metrics on both domains
        """
        with torch.no_grad():
            # Source performance
            source_output = self.source_model(source_inputs.to(self.device))
            source_pred = torch.argmax(source_output, dim=-1)
            source_acc = (source_pred == source_labels.to(self.device)).float().mean().item()

            # Target performance
            target_output = self.target_model(target_inputs.to(self.device))
            target_pred = torch.argmax(target_output, dim=-1)
            target_acc = (target_pred == target_labels.to(self.device)).float().mean().item()

        return {
            'source_accuracy': source_acc,
            'target_accuracy': target_acc,
            'transfer_ratio': target_acc / (source_acc + 1e-8),
        }


class DomainAdaptationTrainer:
    """
    Trains models with domain adaptation techniques.
    Helps models generalize across different game domains.
    """

    def __init__(
        self,
        model: nn.Module,
        domain_discriminator: Optional[nn.Module] = None,
        num_domains: int = 2,
        device: str = 'cpu',
    ):
        """
        Initialize domain adaptation trainer.

        Args:
            model: Feature extractor + task predictor
            domain_discriminator: Optional adversarial discriminator
            num_domains: Number of domains
            device: Torch device
        """
        self.model = model
        self.domain_discriminator = domain_discriminator
        self.num_domains = num_domains
        self.device = device

        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        if domain_discriminator is not None:
            self.discriminator_optimizer = torch.optim.Adam(
                domain_discriminator.parameters(),
                lr=1e-4
            )

        self.training_stats = {
            'task_losses': [],
            'domain_losses': [],
            'domain_accuracies': [],
        }

    def train_step_adversarial(
        self,
        source_data: Tuple[torch.Tensor, torch.Tensor],
        target_data: Tuple[torch.Tensor, torch.Tensor],
        task_loss_fn: Any,
        num_disc_steps: int = 5,
    ) -> Dict[str, float]:
        """
        Train with adversarial domain adaptation.

        Args:
            source_data: (inputs, labels) from source domain
            target_data: (inputs, labels) from target domain
            task_loss_fn: Task loss function
            num_disc_steps: Number of discriminator update steps

        Returns:
            Loss statistics
        """
        source_inputs, source_labels = source_data
        target_inputs, target_labels = target_data

        # Concatenate data
        all_inputs = torch.cat([source_inputs, target_inputs], dim=0)
        domain_labels = torch.cat([
            torch.zeros(source_inputs.shape[0]),
            torch.ones(target_inputs.shape[0])
        ]).long().to(self.device)

        # Get features
        features = self.model(all_inputs.to(self.device))

        # Task loss (on source data only)
        source_features = features[:source_inputs.shape[0]]
        task_loss = task_loss_fn(source_features, source_labels.to(self.device))

        # Domain loss (adversarial)
        domain_pred = self.domain_discriminator(features)
        domain_loss = torch.nn.functional.cross_entropy(domain_pred, domain_labels)

        # Update model (minimize task loss, maximize domain loss)
        combined_loss = task_loss - 0.1 * domain_loss

        self.model_optimizer.zero_grad()
        combined_loss.backward()
        self.model_optimizer.step()

        # Update discriminator
        for _ in range(num_disc_steps):
            features = self.model(all_inputs.to(self.device)).detach()
            domain_pred = self.domain_discriminator(features)
            disc_loss = torch.nn.functional.cross_entropy(domain_pred, domain_labels)

            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            self.discriminator_optimizer.step()

        return {
            'task_loss': task_loss.item(),
            'domain_loss': domain_loss.item(),
        }


if __name__ == "__main__":
    print("Testing Transfer Learning Components...")

    # Test Progressive Neural Network
    print("\n1. Testing Progressive Neural Network...")
    pnn = ProgressiveNeuralNetwork(
        input_dim=10,
        output_dim=6,
        hidden_dims=[128, 128],
        num_tasks=2,
        use_lateral=True
    )

    x = torch.randn(32, 10)
    output_task1 = pnn(x, task_id=0)
    output_task2 = pnn(x, task_id=1)

    print(f"Task 1 output shape: {output_task1.shape}")
    print(f"Task 2 output shape: {output_task2.shape}")

    # Test TransferLearner
    print("\n2. Testing TransferLearner...")
    source_model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 6)
    )
    target_model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 6)
    )

    transfer_learner = TransferLearner(
        source_model, target_model,
        transfer_strategy='fine_tuning'
    )

    print("Transfer learner initialized")

    print("\nAll transfer learning tests passed!")
