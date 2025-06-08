import argparse
import numpy as np
from nanodt.agent import NanoDTAgent
import gymnasium as gym

def evaluate(path_to_model: str, n_episodes: int):
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

    print("\nEvaluation Results:")
    print(f"Mean reward:   {np.mean(rewards):.2f}")
    print(f"Std reward:    {np.std(rewards):.2f}")
    print(f"Min reward:    {np.min(rewards):.2f}")
    print(f"Max reward:    {np.max(rewards):.2f}")
    print(f"Median reward: {np.median(rewards):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NanoDT agent on LunarLanderContinuous")
    parser.add_argument("--path_to_model", type=str, required=True, help="Path to the trained NanoDT model")
    parser.add_argument("--n_episodes", type=int, default=100, help="Number of evaluation episodes")

    args = parser.parse_args()
    evaluate(args.path_to_model, args.n_episodes)
