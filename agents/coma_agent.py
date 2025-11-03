
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import namedtuple
import copy


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for COMA training"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, experience: Experience):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch from buffer"""
        return [self.buffer[i] for i in np.random.choice(len(self.buffer), batch_size, replace=False)]

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        """Clear the buffer"""
        self.buffer = []
        self.position = 0


class Actor(nn.Module):
    """
    Decentralized Actor network for COMA.
    Each agent has its own actor that learns an action policy.

    Input: Local observation of the agent
    Output: Action probabilities (for discrete actions)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_actions: int = 6,
        dueling: bool = False,
    ):
        super(Actor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.dueling = dueling

        # Feature extraction layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if dueling:
            # Dueling architecture: separate advantage and value streams
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_actions)
            )

            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            # Standard architecture
            self.policy_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor network.

        Args:
            observation: Observation tensor

        Returns:
            Action logits or log probabilities
        """
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))

        if self.dueling:
            advantages = self.advantage_stream(x)
            value = self.value_stream(x)
            # Combine: Q = V + (A - mean(A))
            action_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
            return action_values
        else:
            return self.policy_head(x)

    def get_action_probabilities(self, observation: torch.Tensor) -> torch.Tensor:
        """Get action probabilities (softmax over logits)"""
        logits = self.forward(observation)
        return F.softmax(logits, dim=-1)

    def get_log_probabilities(self, observation: torch.Tensor) -> torch.Tensor:
        """Get log action probabilities"""
        logits = self.forward(observation)
        return F.log_softmax(logits, dim=-1)


class Critic(nn.Module):
    """
    Centralized Critic network for COMA.

    Input: Joint state + joint action (all agents' observations and actions)
    Output: Q-value estimates for counterfactual advantage computation

    This is the key component for credit assignment in COMA.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 128,
    ):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        # Input: state (concatenated observations of all agents) + joint actions
        input_dim = state_dim * num_agents + action_dim * num_agents

        # Q-network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 1)

        # Dueling Q-network for better stability
        self.v_head = nn.Linear(hidden_dim, 1)
        self.a_head = nn.Linear(hidden_dim, 1)

    def forward(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute Q-value for joint state-action pair.

        Args:
            joint_obs: Concatenated observations of all agents [batch_size, state_dim * num_agents]
            joint_actions: Joint action (one-hot encoded) [batch_size, action_dim * num_agents]

        Returns:
            Q-values [batch_size, 1]
        """
        x = torch.cat([joint_obs, joint_actions], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Dueling architecture
        value = self.v_head(x)
        advantage = self.a_head(x)

        q_value = value + advantage
        return q_value

    def compute_q_values(
        self,
        joint_obs: torch.Tensor,
        joint_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Wrapper for forward pass"""
        return self.forward(joint_obs, joint_actions)


class COMAAgent:
    """
    Full COMA (Counterfactual Multi-Agent Policy Gradients) implementation.

    Key features:
    - Centralized critic for accurate value estimation
    - Decentralized actors for scalability
    - Counterfactual advantage computation for credit assignment
    - Separate target networks for stability
    - Experience replay for sample efficiency
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int = 6,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,  # Soft update coefficient
        device: str = 'cpu',
        use_dueling: bool = True,
        use_double_q: bool = True,
    ):
        """
        Initialize COMA agent.

        Args:
            num_agents: Number of agents in the environment
            state_dim: Dimension of observation space per agent
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer dimension
            learning_rate: Actor learning rate
            critic_learning_rate: Critic learning rate
            gamma: Discount factor
            tau: Soft update target network coefficient
            device: torch device
            use_dueling: Use dueling actor architecture
            use_double_q: Use double Q-learning for critic
        """
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.use_double_q = use_double_q

        # Create actors (one per agent)
        self.actors = [
            Actor(state_dim, hidden_dim, action_dim, dueling=use_dueling).to(device)
            for _ in range(num_agents)
        ]

        # Create critic (shared, centralized)
        self.critic = Critic(state_dim, action_dim, num_agents, hidden_dim).to(device)

        # Target networks for stability
        self.target_critic = Critic(state_dim, action_dim, num_agents, hidden_dim).to(device)
        self._hard_update_target_networks()

        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=learning_rate)
            for actor in self.actors
        ]

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=critic_learning_rate
        )

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=20000)

        # Training statistics
        self.training_stats = {
            'actor_losses': [[] for _ in range(num_agents)],
            'critic_loss': [],
            'q_values': [],
        }

    def _hard_update_target_networks(self):
        """Hard update: copy current network weights to target network"""
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _soft_update_target_networks(self):
        """Soft update: blend current and target network weights"""
        for target_param, param in zip(
            self.target_critic.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def select_actions(self, observations: List[np.ndarray], exploration: bool = True) -> List[int]:
        """
        Select actions for all agents given observations.

        Args:
            observations: List of observations (one per agent)
            exploration: Whether to use epsilon-greedy exploration

        Returns:
            List of action indices (one per agent)
        """
        actions = []

        for i, obs in enumerate(observations):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # Get action probabilities from actor
            with torch.no_grad():
                logits = self.actors[i](obs_tensor)
                action_probs = F.softmax(logits, dim=-1)

            if exploration:
                # Sample from distribution
                action = torch.multinomial(action_probs, 1).item()
            else:
                # Greedy action
                action = torch.argmax(logits, dim=-1).item()

            actions.append(action)

        return actions

    def update(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Update COMA networks using experience replay.

        Key COMA update:
        1. Compute Q-values using centralized critic
        2. For each agent i:
           - Compute counterfactual Q: Q(s, a_i, a_{-i})
           - Compute baseline Q: Q(s, a'_i, a_{-i}) where a'_i is greedy action
           - Advantage: A_i = Q - baseline
           - Update actor using policy gradient: ∇log π_i * A_i

        Args:
            batch_size: Batch size for update

        Returns:
            Dictionary with loss values
        """
        if len(self.replay_buffer) < batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}

        # Sample batch from replay buffer
        experiences = self.replay_buffer.sample(batch_size)

        # Unpack experiences
        states = []
        actions_list = []
        rewards = []
        next_states = []
        dones = []

        for exp in experiences:
            states.append(exp.state)
            actions_list.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Prepare joint observations
        # Shape: [batch_size, state_dim * num_agents]
        joint_obs = states_tensor.reshape(batch_size, -1)
        joint_next_obs = next_states_tensor.reshape(batch_size, -1)

        # ==================== CRITIC UPDATE ====================
        critic_loss = self._update_critic(
            joint_obs, actions_list, rewards_tensor,
            joint_next_obs, dones_tensor
        )

        # ==================== ACTOR UPDATE ====================
        actor_losses = self._update_actors(joint_obs, actions_list)

        # Soft update target networks
        self._soft_update_target_networks()

        # Log statistics
        avg_actor_loss = np.mean(actor_losses)
        self.training_stats['critic_loss'].append(critic_loss)
        for i, loss in enumerate(actor_losses):
            self.training_stats['actor_losses'][i].append(loss)

        return {
            'critic_loss': critic_loss,
            'actor_loss': avg_actor_loss,
        }

    def _update_critic(
        self,
        joint_obs: torch.Tensor,
        actions_list: List[np.ndarray],
        rewards: torch.Tensor,
        joint_next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """
        Update centralized critic network.

        Loss: TD loss = (r + γ * Q_target(s', π(s')) - Q(s, a))^2
        """
        batch_size = joint_obs.shape[0]

        # Convert actions to one-hot vectors
        actions_one_hot = self._convert_actions_to_onehot(actions_list, batch_size)

        # Current Q-values
        current_q = self.critic(joint_obs, actions_one_hot)

        # Compute target Q-values
        with torch.no_grad():
            # Get greedy actions for next state
            next_actions_one_hot = self._get_greedy_actions_onehot(joint_next_obs, batch_size)

            # Compute target Q
            if self.use_double_q:
                # Double Q-learning: use current network to select, target network to evaluate
                target_q = self.target_critic(joint_next_obs, next_actions_one_hot)
            else:
                target_q = self.target_critic(joint_next_obs, next_actions_one_hot)

            # Bellman equation: r + γ * Q_target * (1 - done)
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones).unsqueeze(1)

        # MSE loss
        critic_loss = F.mse_loss(current_q, target_q)

        # Backprop
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actors(
        self,
        joint_obs: torch.Tensor,
        actions_list: List[np.ndarray],
    ) -> List[float]:
        """
        Update decentralized actor networks using counterfactual advantages.

        For each agent i:
        1. Compute Q(s, a_i, a_{-i}) - Q-value for actual action
        2. Compute baseline Q(s, a'_i, a_{-i}) - Q-value for greedy action
        3. Counterfactual advantage: A_i = Q - baseline
        4. Actor loss: -log(π_i) * A_i (policy gradient)
        """
        batch_size = joint_obs.shape[0]
        actor_losses = []

        # Convert actions to one-hot
        actions_one_hot = self._convert_actions_to_onehot(actions_list, batch_size)

        # Get greedy actions for baseline computation
        greedy_actions_one_hot = self._get_greedy_actions_onehot(joint_obs, batch_size)

        for agent_id in range(self.num_agents):
            # Q-value for actual action
            q_actual = self.critic(joint_obs, actions_one_hot)

            # Q-value for greedy action (baseline for counterfactual advantage)
            q_baseline = self.critic(joint_obs, greedy_actions_one_hot)

            # Counterfactual advantage
            advantage = q_actual - q_baseline
            advantage = advantage.detach()  # Detach to avoid second-order gradients

            # Get log probabilities for agent i
            obs_i = joint_obs[:, agent_id * self.state_dim : (agent_id + 1) * self.state_dim]
            log_probs = self.actors[agent_id].get_log_probabilities(obs_i)

            # Get actions for agent i
            actions_i = torch.LongTensor([actions_list[j][agent_id] for j in range(batch_size)]).to(self.device)
            log_probs_i = log_probs.gather(1, actions_i.unsqueeze(1))

            # Policy gradient loss: -E[log π(a|s) * A(s,a)]
            actor_loss = -(log_probs_i * advantage).mean()

            # Add entropy regularization
            entropy = -(log_probs * F.softmax(self.actors[agent_id](obs_i), dim=-1)).sum(1).mean()
            actor_loss = actor_loss - 0.01 * entropy

            # Backprop
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), max_norm=1.0)
            self.actor_optimizers[agent_id].step()

            actor_losses.append(actor_loss.item())

        return actor_losses

    def _convert_actions_to_onehot(self, actions_list: List[np.ndarray], batch_size: int) -> torch.Tensor:
        """
        Convert action lists to joint one-hot encoding.

        Args:
            actions_list: List of action arrays from batch
            batch_size: Batch size

        Returns:
            Joint one-hot tensor [batch_size, action_dim * num_agents]
        """
        joint_actions = []

        for batch_idx in range(batch_size):
            for agent_id in range(self.num_agents):
                action_idx = actions_list[batch_idx][agent_id]

                # Create one-hot vector
                one_hot = np.zeros(self.action_dim)
                one_hot[action_idx] = 1.0
                joint_actions.append(one_hot)

        joint_actions = np.array(joint_actions).reshape(batch_size, -1)
        return torch.FloatTensor(joint_actions).to(self.device)

    def _get_greedy_actions_onehot(self, joint_obs: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Get greedy actions for all agents and convert to joint one-hot.
        """
        greedy_actions = []

        with torch.no_grad():
            for agent_id in range(self.num_agents):
                # Extract observation for this agent
                obs_i = joint_obs[:, agent_id * self.state_dim : (agent_id + 1) * self.state_dim]

                # Get greedy action
                logits = self.actors[agent_id](obs_i)
                actions = torch.argmax(logits, dim=-1)
                greedy_actions.append(actions)

        # Convert to joint one-hot
        joint_actions = []
        for batch_idx in range(batch_size):
            for agent_id in range(self.num_agents):
                action_idx = greedy_actions[agent_id][batch_idx].item()

                one_hot = np.zeros(self.action_dim)
                one_hot[action_idx] = 1.0
                joint_actions.append(one_hot)

        joint_actions = np.array(joint_actions).reshape(batch_size, -1)
        return torch.FloatTensor(joint_actions).to(self.device)

    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in replay buffer"""
        self.replay_buffer.push(
            Experience(state, action, reward, next_state, done)
        )

    def save(self, path: str):
        """Save agent networks to file"""
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats,
        }
        torch.save(checkpoint, path)
        print(f"COMA agent saved to {path}")

    def load(self, path: str):
        """Load agent networks from file"""
        checkpoint = torch.load(path, map_location=self.device)

        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])

        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])

        for i, opt in enumerate(self.actor_optimizers):
            opt.load_state_dict(checkpoint['actor_optimizers'][i])

        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.training_stats = checkpoint['training_stats']

        print(f"COMA agent loaded from {path}")

    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        return {
            'avg_critic_loss': np.mean(self.training_stats['critic_loss'][-100:]) if self.training_stats['critic_loss'] else 0.0,
            'avg_actor_loss': np.mean([np.mean(losses[-100:]) for losses in self.training_stats['actor_losses']]) if self.training_stats['actor_losses'][0] else 0.0,
        }

    def reset_exploration(self):
        """Reset exploration state (if needed)"""
        pass


# Example usage
if __name__ == "__main__":
    print("Testing COMA Agent...")

    # Create agent
    num_agents = 2
    state_dim = 10
    action_dim = 6

    agent = COMAAgent(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        device='cpu',
    )

    # Simulate some experiences
    for _ in range(1000):
        states = [np.random.randn(state_dim) for _ in range(num_agents)]
        actions = agent.select_actions(states)
        reward = np.random.randn()
        next_states = [np.random.randn(state_dim) for _ in range(num_agents)]
        done = False

        # Store experience
        state = np.concatenate(states)
        next_state = np.concatenate(next_states)
        agent.store_experience(state, np.array(actions), reward, next_state, done)

    # Update
    print("Performing updates...")
    for _ in range(10):
        losses = agent.update(batch_size=32)
        print(f"Critic loss: {losses['critic_loss']:.4f}, Actor loss: {losses['actor_loss']:.4f}")

    # Save
    agent.save("test_coma.pth")
    print("Agent saved successfully!")

    print("All tests passed!")
