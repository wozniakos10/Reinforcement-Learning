import argparse
import gymnasium as gym
from stable_baselines3 import SAC, PPO, DDPG
import os

def main(path_to_save_model, path_to_log, algo):
    # Create environment
    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")

    if algo == "SAC":
    # Instantiate the agent
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=path_to_log)

    elif algo == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=path_to_log)

    elif algo == "DDPG":
        model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=path_to_log)

    # Train the agent
    model.learn(total_timesteps=int(2e6), progress_bar=True)

    # Save the trained model
    os.makedirs(os.path.dirname(path_to_save_model), exist_ok=True)
    model.save(path_to_save_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SAC agent on LunarLanderContinuous")
    parser.add_argument("--path_to_save_model", type=str, required=True, help="Path to save the model")
    parser.add_argument("--path_to_log", type=str, required=True, help="Path to save tensorboard logs")
    parser.add_argument("--algo", type=str, default="SAC", choices=["SAC", "DDPG", "PPO"])

    args = parser.parse_args()
    main(args.path_to_save_model, args.path_to_log, args.algo)
