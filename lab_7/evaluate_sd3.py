import argparse
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

def main(path_to_model):
    # Create environment
    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")

    # Load the trained model
    model = SAC.load(path_to_model, env=env)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

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

    args = parser.parse_args()
    main(args.path_to_model)
