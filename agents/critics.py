import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class StateCritic(nn.Module):
    """
    Simple state-value critic.
    Input: State observation
    Output: State value V(s)

    Used for baseline computation in policy gradient methods.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
    ):
        super(StateCritic, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Build network
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]

        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ])

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.

        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]

        Returns:
            State value [batch_size, 1] or [1]
        """
        return self.network(state)

    def compute_value(self, state: np.ndarray) -> float:
        """Compute value for numpy state (inference mode)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = self.forward(state_tensor).squeeze().item()
        return value


class ActionCritic(nn.Module):
    """
    Action-value critic (Q-network).
    Input: State observation + Action (one-hot encoded)
    Output: Q-value Q(s, a)

    Used in off-policy algorithms like DQN, DDPG.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        dueling: bool = True,
    ):
        super(ActionCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dueling = dueling

        # Common feature extraction
        input_dim = state_dim + action_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ])

        self.feature_net = nn.Sequential(*layers)

        if dueling:
            # Dueling architecture: V + (A - mean(A))
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value.

        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor (one-hot) [batch_size, action_dim]

        Returns:
            Q-value [batch_size, 1]
        """
        # Concatenate state and action
        sa = torch.cat([state, action], dim=-1)

        # Feature extraction
        features = self.feature_net(sa)

        if self.dueling:
            # Dueling formula: Q(s,a) = V(s) + (A(s,a) - mean(A))
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            q_value = value + (advantage - advantage.mean())
        else:
            q_value = self.q_head(features)

        return q_value


class CentralizedCritic(nn.Module):
    """
    Centralized critic for multi-agent training.
    Input: Joint state (all agents' observations) + Joint action (all agents' actions)
    Output: Joint Q-value or Team value

    Used in COMA, QMIX for centralized value estimation.
    """

    def __init__(
        self,
        joint_state_dim: int,
        joint_action_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        use_layer_norm: bool = True,
    ):
        super(CentralizedCritic, self).__init__()

        self.joint_state_dim = joint_state_dim
        self.joint_action_dim = joint_action_dim
        self.hidden_dim = hidden_dim

        # Build network
        input_dim = joint_state_dim + joint_action_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))

        for i in range(num_hidden_layers):
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, joint_state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        """
        Compute joint Q-value or team value.

        Args:
            joint_state: Concatenated states of all agents [batch_size, joint_state_dim]
            joint_action: Joint action (concatenated one-hots) [batch_size, joint_action_dim]

        Returns:
            Joint Q-value [batch_size, 1]
        """
        joint_sa = torch.cat([joint_state, joint_action], dim=-1)
        return self.network(joint_sa)


class IndependentCritic(nn.Module):
    """
    Independent critic for each agent (MAAC/MADDPG style).
    Each agent has its own Q-network that sees full state but only its own action.

    This is useful for decentralized execution after centralized training.
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        shared_backbone: bool = False,
    ):
        super(IndependentCritic, self).__init__()

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.shared_backbone = shared_backbone

        if shared_backbone:
            # All agents share feature extraction
            input_dim = state_dim
            self.shared_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            # Each agent has its own action-fusion head
            self.agent_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim + action_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_agents)
            ])
        else:
            # Each agent has its own full Q-network
            input_dim = state_dim + action_dim
            self.agent_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_agents)
            ])

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        agent_id: int,
    ) -> torch.Tensor:
        """
        Compute Q-value for specific agent.

        Args:
            state: Full state observation [batch_size, state_dim]
            action: Action of agent (one-hot) [batch_size, action_dim]
            agent_id: Which agent's Q-network to use

        Returns:
            Q-value [batch_size, 1]
        """
        if self.shared_backbone:
            features = self.shared_net(state)
            sa = torch.cat([features, action], dim=-1)
            q_value = self.agent_heads[agent_id](sa)
        else:
            sa = torch.cat([state, action], dim=-1)
            q_value = self.agent_networks[agent_id](sa)

        return q_value

    def forward_all_agents(
        self,
        state: torch.Tensor,
        actions: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Compute Q-values for all agents.

        Args:
            state: Full state
            actions: List of actions (one per agent)

        Returns:
            List of Q-values (one per agent)
        """
        q_values = []
        for agent_id in range(self.num_agents):
            q = self.forward(state, actions[agent_id], agent_id)
            q_values.append(q)

        return q_values


class AdvantageWeightedCritic(nn.Module):
    """
    Advantage-weighted critic for counterfactual advantage computation.

    Computes Q(s, a_i, a_{-i}) for counterfactual reasoning:
    - What if this agent took different action while others keep their actions?

    This is the key component for COMA's credit assignment.
    """

    def __init__(
        self,
        joint_state_dim: int,
        num_agents: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super(AdvantageWeightedCritic, self).__init__()

        self.joint_state_dim = joint_state_dim
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Total input: joint state + joint action (all agents' actions)
        joint_action_dim = action_dim * num_agents
        input_dim = joint_state_dim + joint_action_dim

        # Build main network for base Q-value
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ])

        self.base_net = nn.Sequential(*layers)

        # Separate heads for:
        # 1. Base Q-value
        # 2. Advantage streams (one per agent)
        self.q_value_head = nn.Linear(hidden_dim, 1)

        self.advantage_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(num_agents)
        ])

    def forward(self, joint_state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value with dueling advantage decomposition.

        Q(s, a) = V(s) + A(s, a) - mean(A(s, a))

        Args:
            joint_state: [batch_size, joint_state_dim]
            joint_action: [batch_size, num_agents * action_dim]

        Returns:
            Q-value [batch_size, 1]
        """
        joint_sa = torch.cat([joint_state, joint_action], dim=-1)
        features = self.base_net(joint_sa)

        # Base Q-value (V(s))
        base_q = self.q_value_head(features)

        # Advantages for all agents
        advantages = []
        for i in range(self.num_agents):
            adv = self.advantage_heads[i](features)
            advantages.append(adv)

        # Stack advantages
        advantages = torch.stack(advantages, dim=1)  # [batch_size, num_agents, 1]

        # Compute mean advantage across agents
        mean_advantage = advantages.mean(dim=1)  # [batch_size, 1]

        # Dueling decomposition: Q = V + (A - mean(A))
        q_value = base_q + (advantages.squeeze(2) - mean_advantage).mean(dim=1, keepdim=True)

        return q_value

    def compute_counterfactual_q(
        self,
        joint_state: torch.Tensor,
        joint_action: torch.Tensor,
        agent_id: int,
        counterfactual_action: int,
    ) -> torch.Tensor:
        """
        Compute counterfactual Q-value: Q(s, a'_i, a_{-i})

        What would the Q-value be if agent i took different action?

        Args:
            joint_state: Joint state [batch_size, joint_state_dim]
            joint_action: Original joint action [batch_size, num_agents * action_dim]
            agent_id: Which agent's action to change
            counterfactual_action: New action for agent i

        Returns:
            Counterfactual Q-value [batch_size, 1]
        """
        batch_size = joint_action.shape[0]

        # Create counterfactual action
        counterfactual_joint_action = joint_action.clone()

        # Extract action portion for agent i
        agent_action_start = agent_id * self.action_dim
        agent_action_end = (agent_id + 1) * self.action_dim

        # Reset agent i's action
        counterfactual_joint_action[:, agent_action_start:agent_action_end] = 0.0

        # Set counterfactual action (one-hot)
        counterfactual_joint_action[:, agent_action_start + counterfactual_action] = 1.0

        # Compute Q with counterfactual action
        return self.forward(joint_state, counterfactual_joint_action)


class MultiHeadCritic(nn.Module):
    """
    Multi-head critic that can output multiple value estimates.
    Useful for:
    - Value and advantage decomposition
    - Uncertainty estimation
    - Multi-task learning
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 3,
        num_hidden_layers: int = 2,
    ):
        super(MultiHeadCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Shared feature extraction
        input_dim = state_dim + action_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ])

        self.shared_net = nn.Sequential(*layers)

        # Multiple Q-heads (ensembles for uncertainty)
        self.q_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(num_heads)
        ])

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple heads.

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]

        Returns:
            Dictionary with:
            - 'q_values': All Q-estimates [batch_size, num_heads]
            - 'q_mean': Mean Q-estimate [batch_size, 1]
            - 'q_std': Std of Q-estimates [batch_size, 1]
        """
        sa = torch.cat([state, action], dim=-1)
        features = self.shared_net(sa)

        # Compute Q-values from all heads
        q_values = []
        for head in self.q_heads:
            q = head(features)
            q_values.append(q)

        q_values = torch.cat(q_values, dim=-1)  # [batch_size, num_heads]

        # Statistics
        q_mean = q_values.mean(dim=-1, keepdim=True)
        q_std = q_values.std(dim=-1, keepdim=True)

        return {
            'q_values': q_values,
            'q_mean': q_mean,
            'q_std': q_std,
        }


# Factory functions for creating critics

def create_state_critic(state_dim: int, hidden_dim: int = 128) -> StateCritic:
    """Create a state value critic"""
    return StateCritic(state_dim, hidden_dim)


def create_action_critic(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 128,
    dueling: bool = True,
) -> ActionCritic:
    """Create an action-value (Q) critic"""
    return ActionCritic(state_dim, action_dim, hidden_dim, dueling=dueling)


def create_centralized_critic(
    num_agents: int,
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
) -> CentralizedCritic:
    """Create a centralized critic for multi-agent training"""
    joint_state_dim = state_dim * num_agents
    joint_action_dim = action_dim * num_agents
    return CentralizedCritic(joint_state_dim, joint_action_dim, hidden_dim)


# Example usage
if __name__ == "__main__":
    print("Testing Critic Networks...")

    # Test StateCritic
    print("\n1. Testing StateCritic...")
    state_critic = StateCritic(state_dim=10, hidden_dim=64)
    state = torch.randn(32, 10)
    value = state_critic(state)
    print(f"State value shape: {value.shape}")  # Should be [32, 1]

    # Test ActionCritic
    print("\n2. Testing ActionCritic...")
    action_critic = ActionCritic(state_dim=10, action_dim=6, hidden_dim=64)
    action = F.one_hot(torch.randint(0, 6, (32,)), num_classes=6).float()
    q_value = action_critic(state, action)
    print(f"Q-value shape: {q_value.shape}")  # Should be [32, 1]

    # Test CentralizedCritic
    print("\n3. Testing CentralizedCritic...")
    num_agents = 2
    state_dim = 10
    action_dim = 6

    centralized_critic = CentralizedCritic(
        joint_state_dim=state_dim * num_agents,
        joint_action_dim=action_dim * num_agents,
        hidden_dim=128,
    )

    joint_state = torch.randn(32, state_dim * num_agents)
    joint_action = torch.randn(32, action_dim * num_agents)
    joint_q = centralized_critic(joint_state, joint_action)
    print(f"Joint Q-value shape: {joint_q.shape}")  # Should be [32, 1]

    # Test IndependentCritic
    print("\n4. Testing IndependentCritic...")
    ind_critic = IndependentCritic(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    q_values = ind_critic.forward_all_agents(joint_state, [action, action])
    print(f"Independent Q-values: {[q.shape for q in q_values]}")  # Should be [32, 1] each

    # Test AdvantageWeightedCritic
    print("\n5. Testing AdvantageWeightedCritic...")
    adv_critic = AdvantageWeightedCritic(
        joint_state_dim=state_dim * num_agents,
        num_agents=num_agents,
        action_dim=action_dim,
    )

    adv_q = adv_critic(joint_state, joint_action)
    print(f"Advantage Q-value shape: {adv_q.shape}")  # Should be [32, 1]

    # Test counterfactual Q
    cf_q = adv_critic.compute_counterfactual_q(joint_state, joint_action, agent_id=0, counterfactual_action=2)
    print(f"Counterfactual Q-value shape: {cf_q.shape}")  # Should be [32, 1]

    # Test MultiHeadCritic
    print("\n6. Testing MultiHeadCritic...")
    multi_critic = MultiHeadCritic(state_dim=10, action_dim=6, num_heads=3)
    output = multi_critic(state, action)
    print(f"Multi-head output - Q values shape: {output['q_values'].shape}")  # [32, 3]
    print(f"Multi-head output - Q mean shape: {output['q_mean'].shape}")  # [32, 1]
    print(f"Multi-head output - Q std shape: {output['q_std'].shape}")  # [32, 1]

    print("\nAll critic tests passed!")
