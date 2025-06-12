from argparse import ArgumentParser

import gymnasium as gym
import torch
from stable_baselines3 import DDPG, PPO, SAC
from tqdm.auto import tqdm

from minari import DataCollector


def collect_dataset(env_name: str, expert_path: str, n_timesteps: int, seed: int, algo):
    torch.manual_seed(seed)
    env = DataCollector(gym.make(env_name, continuous=True))
    if algo == "SAC":
        # Load the trained model
        agent = SAC.load(expert_path, env=env)

    elif algo == "PPO":
        agent = PPO.load(expert_path, env=env)

    elif algo == "DDPG":
        agent = DDPG.load(expert_path, env=env)

    done = True

    for i in tqdm(range(n_timesteps)):
        if done:
            obs, _ = env.reset(seed=seed)

        action, _ = agent.predict(obs)
        obs, rew, ter, tru, info = env.step(action)

        done = ter or tru

    dataset = env.create_dataset(
        dataset_id=f"{env_name}-ppo-2e6-v0",
        algorithm_name="ExpertPolicy",
        # code_permalink="https://minari.farama.org/tutorials/behavioral_cloning",
        author="Dawid Wozniak",
        author_email="wozniakos10@gmail.com",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env-name", type=str, default="LunarLander-v3")
    parser.add_argument(
        "--expert-path", type=str, default="models/sac_lunar_continuous_5e5.zip"
    )
    parser.add_argument("--n-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--algo", type=str, default="SAC", choices=["SAC", "DDPG", "PPO"])

    args = parser.parse_args()

    collect_dataset(
        env_name=args.env_name,
        expert_path=args.expert_path,
        n_timesteps=args.n_timesteps,
        seed=args.seed,
        algo = args.algo)