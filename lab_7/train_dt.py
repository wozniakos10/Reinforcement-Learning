import argparse
import pickle

import minari

from nanodt.agent import NanoDTAgent
from nanodt.utils import seed_libraries


def train_dt(dataset_id: str, path_to_save: str):
    seed = 1234
    seed_libraries(seed)
    minari_dataset = minari.load_dataset(dataset_id)

    dt_agent = NanoDTAgent(device="mps")
    dt_agent.learn(minari_dataset, reward_scale=1000.0,
                   #max_iters=1000
                   )
    dt_agent.save(path_to_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Decision Transformer agent.")
    parser.add_argument("--dataset_id", type=str, required=True, help="Minari dataset ID to load")
    parser.add_argument("--path_to_save", type=str, required=True, help="Path to save the trained model")

    args = parser.parse_args()
    train_dt(args.dataset_id, args.path_to_save)
