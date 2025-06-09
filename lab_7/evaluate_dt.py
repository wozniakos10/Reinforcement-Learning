import argparse
import numpy as np
from nanodt.agent import NanoDTAgent
import gymnasium as gym
import json

def evaluate(path_to_model: str, n_episodes: int, path_to_save: str):
    env = gym.make("LunarLander-v3", continuous=True)
    agent = NanoDTAgent.load(path_to_model)
    rewards = []

    for i in range(n_episodes):
        agent.reset(target_return=6000)
        obs, info = env.reset()
        done = False
        accumulated_rew = 0

        while not done:
            action = agent.act(obs)
            obs, rew, ter, tru, info = env.step(action)
            done = ter or tru
            accumulated_rew += rew

        if i % 100 == 0 or i == n_episodes - 1:
            print(f"Successfully processed {i+1} episodes")

        rewards.append(accumulated_rew)

    mean = np.mean(rewards)
    std = np.std(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)

    print(f"---------stats---------")
    print(f"Mean reward: {mean}")
    print(f"Max reward: {max_reward}")
    print(f"Min reward: {max_reward}")
    print(f"Std reward: {std}")
    print(f"---------stats---------")


    stats = {
        "mean_reward": mean,
        "max_reward": max_reward,
        "min_reward": min_reward,
        "std_reward": std,
        "rewards": rewards
    }

    with open(path_to_save, "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NanoDT agent on LunarLanderContinuous")
    parser.add_argument("--path_to_model", type=str, required=True, help="Path to the trained NanoDT model")
    parser.add_argument("--n_episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--path_to_save", type=str, required=True, help="Path to the saved model")


    args = parser.parse_args()
    evaluate(args.path_to_model, args.n_episodes, args.path_to_save)
