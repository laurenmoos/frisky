from argparse import ArgumentParser
from pytorch_lightning import Trainer
from networks import create_mlp, ActorCategorical, ActorCriticAgent
import gym
from proximal_policy_optimization import RiskAwarePPO

try:
    import gym
except ModuleNotFoundError:
    _GYM_AVAILABLE = False
else:
    _GYM_AVAILABLE = True


def main(hparams):

    env = gym.make(hparams.env_str)
    obs_dim = env.observation_space.shape
    num_actions = env.action_space.n

    actor_network = create_mlp(obs_dim, num_actions, list(hparams.hidden_sizes))
    actor = ActorCategorical(actor_network)

    critic_network = create_mlp(obs_dim, 1, list(hparams.hidden_sizes))
    agent = ActorCriticAgent(actor, critic_network)
    if not _GYM_AVAILABLE:
        raise ModuleNotFoundError('This Module requires gym environment which is not installed yet.')

    model = RiskAwarePPO(
        env,
        agent,
        hparams.epochs,
        hparams.steps_per_epoch,
        hparams.tr_policy_iter,
        hparams.tr_value_iter,
        hparams.learning_rate,
        hparams.risk_aware,
        hparams.value_loss_coef,
        hparams.entropy_beta,
        hparams.clip_ratio)
    trainer = Trainer(accelerator=hparams.accelerator, devices=hparams.devices, max_epochs=hparams.epochs)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", default=(3e-4, 1e-3))
    parser.add_argument("--epochs", default=30)
    parser.add_argument("--steps_per_epoch", default=4000)
    parser.add_argument("---tr_policy_iter", default=80)
    parser.add_argument("---tr_value_iter", default=80)
    parser.add_argument("---hidden_sizes", default=(64, 64))
    parser.add_argument("--risk_aware", default=False)
    parser.add_argument("--value_loss_coef", default=0.99)
    parser.add_argument("--entropy_beta", default=0.95)
    parser.add_argument("--clip_ratio", default=0.2)
    parser.add_argument("--env_str", default='CartPole-v1')
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)
