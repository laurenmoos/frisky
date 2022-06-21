
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from typing import Union, Tuple


class AtariModel(nn.Module):
    def __init__(self):
        super().__init__()

        # The first convolution layer takes a
        # 84x84 frame and produces a 20x20 frame
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)

        # The second convolution layer takes a
        # 20x20 frame and produces a 9x9 frame
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)

        # The third convolution layer takes a
        # 9x9 frame and produces a 7x7 frame
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # A fully connected layer takes the flattened
        # frame from third convolution layer, and outputs
        # 512 features
        self.lin = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        # A fully connected layer to get logits for $\pi$
        self.pi_logits = nn.Linear(in_features=512, out_features=4)

        # A fully connected layer to get value function
        self.value = nn.Linear(in_features=512, out_features=1)

        #
        self.activation = nn.ReLU()

    def __call__(self, obs: torch.Tensor):
        net_layers = [self.activation, self.conv1, self.activation, self.conv2, self.activation, self.conv3,
                      nn.Linear(-1, 7 * 7 * 64)]

        return nn.Sequential(*net_layers)


# note: for discrete action spaces
class ActorCategorical(nn.Module):
    """
    Policy network, for discrete action spaces, which returns a distribution
    and an action given an observation
    """

    def __init__(self, actor_net):
        super().__init__()

        self.actor_net = actor_net

    def forward(self, states):
        logits = self.actor_net(states.float())
        pi = Categorical(logits=logits)
        actions = pi.sample()

        return pi, actions

    def get_log_prob(self, pi: Categorical, actions: torch.Tensor):
        """
        Takes in a distribution and actions and returns log prob of actions
        under the distribution
        Args:
            pi: torch distribution
            actions: actions taken by distribution
        Returns:
            log probability of the acition under pi
        """
        return pi.log_prob(actions)


class ActorCriticAgent(object):
    """
    Actor Critic Agent used during trajectory collection. It returns a
    distribution and an action given an observation. Agent based on the
    implementations found here: https://github.com/Shmuma/ptan/blob/master/ptan/agent.py
    """

    def __init__(self, actor_net: nn.Module, critic_net: nn.Module):
        # TODO: shouldn't this be the same network?
        self.actor_net = actor_net
        self.critic_net = critic_net

    @torch.no_grad()
    def __call__(self, state: torch.Tensor, device: str) -> Tuple:
        """
        Takes in the current state and returns the agents policy, sampled
        action, log probability of the action, and value of the given state
        Args:
            state: current state of the environment
            device: the device used for the current batch
        Returns:
            torch dsitribution and randomly sampled action
        """

        state = state.to(device=device)

        pi, actions = self.actor_net(state)
        log_p = self.get_log_prob(pi, actions)

        value = self.critic_net(state.float())

        return pi, actions, log_p, value

    def get_log_prob(self,
                     pi: Union[Categorical, Normal],
                     actions: torch.Tensor) -> torch.Tensor:
        """
        Takes in the current state and returns the agents policy, a sampled
        action, log probability of the action, and the value of the state
        Args:
            pi: torch distribution
            actions: actions taken by distribution
        Returns:
            log probability of the acition under pi
        """
        return self.actor_net.get_log_prob(pi, actions)
