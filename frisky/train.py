from argparse import ArgumentParser
from pytorch_lightning import Trainer
from networks import AtariModel, ActorCategorical, ActorCriticAgent
import gym
from proximal_policy_optimization import RiskAwarePPO

try:
    import gym
except ModuleNotFoundError:
    _GYM_AVAILABLE = False
else:
    _GYM_AVAILABLE = True


def main(hparams):
    actor_lr, critic_lr = hparams.learning_rate
    actor_network = AtariModel()
    actor = ActorCategorical(actor_network)

    critic_network = AtariModel()
    agent = ActorCriticAgent(actor, critic_network)
    if not _GYM_AVAILABLE:
        raise ModuleNotFoundError('This Module requires gym environment which is not installed yet.')

    env = gym.make(hparams.env_str)
    model = RiskAwarePPO(
        env,
        agent,
        hparams.updates,
        hparams.epochs,
        hparams.batch_size,
        hparams.steps_per_epoch,
        hparams.risk_aware,
        hparams.value_loss_coef,
        hparams.entropy_beta,
        hparams.clip_ratio)
    trainer = Trainer(accelerator=hparams.accelerator, devices=hparams.devices, max_epochs=hparams.updates)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", default=(2.5e-4, 1e-3))
    parser.add_argument("--updates", default=10000)
    parser.add_argument("--epochs", default=8)
    parser.add_argument("--batch_size", default=524)
    parser.add_argument("--steps_per_epoch", default=None)
    parser.add_argument("--risk_aware", default=False)
    parser.add_argument("--value_loss_coef", default=0.5)
    parser.add_argument("--entropy_beta", default=0.01)
    parser.add_argument("--clip_ratio", default=0.1)
    parser.add_argument("--env_str")
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)
