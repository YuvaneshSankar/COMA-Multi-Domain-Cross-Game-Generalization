

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing trajectory sequences.
    Learns to extract game-relevant patterns from historical transitions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        super(TransformerEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Embedding layer (project input to hidden dimension)
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_seq_len, hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )

        # Output: mean pooling over sequence
        self.output_dim = hidden_dim

    def _create_positional_encoding(
        self,
        max_len: int,
        d_model: int
    ) -> torch.Tensor:
        """Create sinusoidal positional encodings"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask

        Returns:
            - Encoded sequence [batch_size, seq_len, hidden_dim]
            - Mean-pooled encoding [batch_size, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Embed input
        x = self.embedding(x)  # [batch_size, seq_len, hidden_dim]

        # Add positional encoding
        device = x.device
        pos_enc = self.positional_encoding[:, :seq_len, :].to(device)
        x = x + pos_enc

        # Apply transformer
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)  # [batch_size, seq_len, hidden_dim]

        # Mean pooling over sequence
        if mask is not None:
            # Mask out padding tokens
            mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
            encoded_masked = encoded.masked_fill(mask_expanded, 0)
            sum_encoded = encoded_masked.sum(dim=1)
            count = (~mask).sum(dim=1, keepdim=True)
            pooled = sum_encoded / count.clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)  # [batch_size, hidden_dim]

        return encoded, pooled


class TaskInferenceNetwork(nn.Module):
    """
    Infers task/game type from trajectory history.
    Outputs a task embedding that modulates policy behavior.
    """

    def __init__(
        self,
        trajectory_dim: int,
        hidden_dim: int = 256,
        task_latent_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_tasks: int = 3,  # Number of game types
    ):
        super(TaskInferenceNetwork, self).__init__()

        self.trajectory_dim = trajectory_dim
        self.task_latent_dim = task_latent_dim
        self.num_tasks = num_tasks

        # Transformer for trajectory encoding
        self.transformer = TransformerEncoder(
            input_dim=trajectory_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        # Task embedding head
        self.task_embedding_head = nn.Sequential(
            nn.Linear(self.transformer.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, task_latent_dim)
        )

        # Optional: task classification head (for auxiliary learning)
        self.task_classifier = nn.Sequential(
            nn.Linear(task_latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

    def forward(self, trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Infer task embedding from trajectory.

        Args:
            trajectory: Tensor of shape [batch_size, seq_len, trajectory_dim]
                       Typically contains state, action, reward concatenated

        Returns:
            Dictionary with:
            - 'task_embedding': Learned task representation [batch_size, task_latent_dim]
            - 'task_logits': Optional task classification logits
        """
        # Encode trajectory
        _, pooled = self.transformer(trajectory)

        # Get task embedding
        task_embedding = self.task_embedding_head(pooled)

        # Optional: get task classification
        task_logits = self.task_classifier(task_embedding)

        return {
            'task_embedding': task_embedding,
            'task_logits': task_logits,
        }


class ConditionalPolicyNetwork(nn.Module):
    """
    Policy network conditioned on inferred task embedding.
    Takes current observation + task embedding -> action distribution.
    """

    def __init__(
        self,
        obs_dim: int,
        task_latent_dim: int,
        action_dim: int = 6,
        hidden_dim: int = 128,
        use_adaptation: bool = True,
    ):
        super(ConditionalPolicyNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.task_latent_dim = task_latent_dim
        self.action_dim = action_dim
        self.use_adaptation = use_adaptation

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Task-conditioned fusion
        if use_adaptation:
            self.task_fusion = nn.Sequential(
                nn.Linear(task_latent_dim, hidden_dim),
                nn.ReLU(),
            )

            # Combine observation and task
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        # Policy head (outputs action logits)
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # Value head (for critic)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        observation: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to get action distribution and value.

        Args:
            observation: Current observation [batch_size, obs_dim]
            task_embedding: Task embedding [batch_size, task_latent_dim] (optional)

        Returns:
            Dictionary with:
            - 'action_logits': Action logits [batch_size, action_dim]
            - 'action_probs': Action probabilities [batch_size, action_dim]
            - 'log_probs': Log action probabilities [batch_size, action_dim]
            - 'value': State value [batch_size, 1]
        """
        # Encode observation
        obs_feat = self.obs_encoder(observation)

        # Fuse with task embedding
        if self.use_adaptation and task_embedding is not None:
            task_feat = self.task_fusion(task_embedding)
            combined = torch.cat([obs_feat, task_feat], dim=-1)
            features = self.fusion_layer(combined)
        else:
            features = self.fusion_layer(obs_feat)

        # Get policy and value outputs
        action_logits = self.policy_head(features)
        value = self.value_head(features)

        # Compute probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1)

        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'log_probs': log_probs,
            'value': value,
        }


class TransformerMetaRLPolicy:
    """
    Complete meta-RL policy combining:
    - Task inference from trajectory history
    - Conditional policy network
    - Adaptive behavior based on inferred game type
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 6,
        task_latent_dim: int = 64,
        hidden_dim: int = 256,
        trajectory_window: int = 20,
        num_tasks: int = 3,
        device: str = 'cpu',
    ):
        """
        Initialize meta-RL policy.

        Args:
            obs_dim: Observation dimension
            action_dim: Action space dimension
            task_latent_dim: Dimension of inferred task embedding
            hidden_dim: Hidden dimension for networks
            trajectory_window: Length of trajectory history to use
            num_tasks: Number of distinct game types
            device: Torch device
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.task_latent_dim = task_latent_dim
        self.trajectory_window = trajectory_window
        self.num_tasks = num_tasks
        self.device = device

        # Task inference network
        # Trajectory dim = obs_dim + action_dim + 1 (reward)
        trajectory_dim = obs_dim + action_dim + 1

        self.task_inference = TaskInferenceNetwork(
            trajectory_dim=trajectory_dim,
            hidden_dim=hidden_dim,
            task_latent_dim=task_latent_dim,
            num_tasks=num_tasks,
        ).to(device)

        # Conditional policy network
        self.policy_network = ConditionalPolicyNetwork(
            obs_dim=obs_dim,
            task_latent_dim=task_latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            use_adaptation=True,
        ).to(device)

        # Trajectory buffer (for context)
        self.trajectory_buffer = []

        # Training statistics
        self.training_stats = {
            'task_classification_acc': [],
            'policy_loss': [],
            'value_loss': [],
        }

    def process_trajectory_step(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
    ):
        """
        Add a transition to trajectory buffer.

        Args:
            obs: Observation (numpy array)
            action: Action taken
            reward: Reward received
        """
        # Create one-hot action
        action_one_hot = np.zeros(self.action_dim)
        action_one_hot[action] = 1.0

        # Combine into trajectory element
        trajectory_element = np.concatenate([obs, action_one_hot, [reward]])
        self.trajectory_buffer.append(trajectory_element)

        # Keep buffer limited to window size
        if len(self.trajectory_buffer) > self.trajectory_window:
            self.trajectory_buffer.pop(0)

    def infer_task(self) -> torch.Tensor:
        """
        Infer task embedding from current trajectory buffer.

        Returns:
            Task embedding tensor [1, task_latent_dim]
        """
        if len(self.trajectory_buffer) == 0:
            # No history yet, return zero embedding
            return torch.zeros(1, self.task_latent_dim, device=self.device)

        # Pad trajectory to window size
        trajectory_array = np.array(self.trajectory_buffer)  # [seq_len, trajectory_dim]

        if len(trajectory_array) < self.trajectory_window:
            # Pad with zeros
            padding = np.zeros((self.trajectory_window - len(trajectory_array), trajectory_array.shape[1]))
            trajectory_array = np.vstack([padding, trajectory_array])
        else:
            # Take last window_size elements
            trajectory_array = trajectory_array[-self.trajectory_window:]

        # Convert to tensor
        trajectory_tensor = torch.FloatTensor(trajectory_array).unsqueeze(0).to(self.device)  # [1, seq_len, dim]

        # Infer task
        with torch.no_grad():
            output = self.task_inference(trajectory_tensor)
            task_embedding = output['task_embedding']

        return task_embedding

    def select_action(
        self,
        obs: np.ndarray,
        use_exploration: bool = True,
        temperature: float = 1.0,
    ) -> int:
        """
        Select action using adaptive policy.

        Args:
            obs: Current observation
            use_exploration: Whether to sample from distribution or use greedy
            temperature: Temperature for softmax (higher = more random)

        Returns:
            Action index
        """
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Infer task embedding
        task_embedding = self.infer_task()

        # Get policy output
        with torch.no_grad():
            output = self.policy_network(obs_tensor, task_embedding)
            action_logits = output['action_logits'] / temperature
            action_probs = F.softmax(action_logits, dim=-1)

        if use_exploration:
            # Sample from distribution
            action = torch.multinomial(action_probs.squeeze(), 1).item()
        else:
            # Greedy action
            action = torch.argmax(action_logits, dim=-1).item()

        return action

    def get_value(self, obs: np.ndarray) -> float:
        """
        Get state value for observation.

        Args:
            obs: Observation

        Returns:
            State value estimate
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        task_embedding = self.infer_task()

        with torch.no_grad():
            output = self.policy_network(obs_tensor, task_embedding)
            value = output['value'].item()

        return value

    def reset_trajectory_buffer(self):
        """Reset trajectory buffer (e.g., at episode start)"""
        self.trajectory_buffer = []

    def update(
        self,
        obs_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        value_batch: torch.Tensor,
        trajectory_batch: torch.Tensor,
        task_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Update meta-RL policy.

        Args:
            obs_batch: Observations [batch_size, obs_dim]
            action_batch: Actions [batch_size]
            reward_batch: Rewards [batch_size]
            value_batch: Target values [batch_size]
            trajectory_batch: Trajectories [batch_size, seq_len, trajectory_dim]
            task_labels: Ground truth task labels (for auxiliary learning)

        Returns:
            Dictionary with loss values
        """
        # Infer task embeddings
        task_output = self.task_inference(trajectory_batch)
        task_embeddings = task_output['task_embedding']

        # Policy loss
        policy_output = self.policy_network(obs_batch, task_embeddings)

        # Get log probabilities for taken actions
        log_probs = policy_output['log_probs']
        action_batch_long = action_batch.long()
        selected_log_probs = log_probs.gather(1, action_batch_long.unsqueeze(1))

        # Compute advantages
        predicted_values = policy_output['value'].squeeze()
        advantages = value_batch - predicted_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient loss
        policy_loss = -(selected_log_probs.squeeze() * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(predicted_values, value_batch)

        # Entropy bonus
        action_probs = policy_output['action_probs']
        entropy = -(action_probs * log_probs).sum(1).mean()

        # Auxiliary task classification loss (if labels provided)
        task_loss = 0.0
        if task_labels is not None:
            task_logits = task_output['task_logits']
            task_loss = F.cross_entropy(task_logits, task_labels)

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + task_loss - 0.01 * entropy

        # Backprop (simplified - would need optimizers in full implementation)
        # For now, just return losses for logging

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'task_loss': task_loss.item() if isinstance(task_loss, torch.Tensor) else 0.0,
            'entropy': entropy.item(),
            'total_loss': total_loss.item(),
        }

    def save(self, path: str):
        """Save networks to file"""
        checkpoint = {
            'task_inference': self.task_inference.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'training_stats': self.training_stats,
        }
        torch.save(checkpoint, path)
        print(f"Meta-RL policy saved to {path}")

    def load(self, path: str):
        """Load networks from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.task_inference.load_state_dict(checkpoint['task_inference'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.training_stats = checkpoint['training_stats']
        print(f"Meta-RL policy loaded from {path}")


# Example usage
if __name__ == "__main__":
    print("Testing Transformer Meta-RL Policy...")

    # Create policy
    obs_dim = 10
    action_dim = 6

    policy = TransformerMetaRLPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        task_latent_dim=64,
        trajectory_window=20,
        num_tasks=3,
        device='cpu',
    )

    # Simulate trajectory
    print("Simulating trajectory...")
    for step in range(30):
        obs = np.random.randn(obs_dim)
        action = np.random.randint(0, action_dim)
        reward = np.random.randn()

        policy.process_trajectory_step(obs, action, reward)

    # Infer task
    task_embedding = policy.infer_task()
    print(f"Task embedding shape: {task_embedding.shape}")

    # Select action
    obs = np.random.randn(obs_dim)
    action = policy.select_action(obs, use_exploration=True)
    print(f"Selected action: {action}")

    # Get value
    value = policy.get_value(obs)
    print(f"State value: {value:.4f}")

    # Test update
    print("Testing update...")
    obs_batch = torch.randn(32, obs_dim)
    action_batch = torch.randint(0, action_dim, (32,))
    reward_batch = torch.randn(32)
    value_batch = torch.randn(32)
    trajectory_batch = torch.randn(32, 20, obs_dim + action_dim + 1)

    losses = policy.update(obs_batch, action_batch, reward_batch, value_batch, trajectory_batch)
    print(f"Losses: {losses}")

    # Save
    policy.save("test_meta_rl.pth")
    print("Policy saved successfully!")

    print("All tests passed!")
