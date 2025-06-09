import argparse
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import json

def main(path_to_model: str, path_to_save: str):
    # Create environment
    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")

    # Load the trained model
    model = SAC.load(path_to_model, env=env)

    # Evaluate the agent
    rewards, episodes_length = evaluate_policy(model, model.get_env(), n_eval_episodes=1000, return_episode_rewards=True)
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

    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC agent on LunarLanderContinuous")
    parser.add_argument("--path_to_model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--path_to_save", type=str, required=True, help="Path to the saved model")

    args = parser.parse_args()
    main(args.path_to_model, args.path_to_save)
