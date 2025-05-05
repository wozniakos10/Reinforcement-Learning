from solution import Experiment, Corner, OffPolicyNStepSarsaDriver, Environment
import os
import json
from logger import configure_logger
import time


def benchmark() -> None:
    array_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    logger = configure_logger(f"sarsa_benchmark_logger_{array_id}.logs")
    possible_alfa_values_dict = {
        1: [0.01],
        2: [0.05],
        3: [0.1],
        4: [0.2],
        5: [0.3],
        6: [0.4],
        7: [0.5],
        8: [0.6],
        9: [0.7],
        10: [0.8],
        11: [0.9],
        12: [1],
    }

    possible_alfa_values = possible_alfa_values_dict.get(array_id)
    possible_n_values = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    for alfa in possible_alfa_values:
        for n_value in possible_n_values:
            start_time = time.time()
            logger.info(f"alfa = {alfa}, n_value = {n_value}")
            experiment = Experiment(
                environment=Environment(
                    corner=Corner(name="corner_c"),
                    steering_fail_chance=0.01,
                ),
                driver=OffPolicyNStepSarsaDriver(
                    step_no=n_value,
                    step_size=alfa,
                    experiment_rate=0.05,
                    discount_factor=1.00,
                ),
                number_of_episodes=10000,
            )

            penalties = experiment.run()

            stop_time = time.time()
            configuration_time = stop_time - start_time
            logger.info(f"Time needed to complete run for current configuration: {configuration_time:5f}")
            if not os.path.exists("benchmarks"):
                os.makedirs("benchmarks")

            with open(f"benchmarks/penalty_alfa_{alfa}_n_value_{n_value}.json", "w") as file:
                json.dump({"penalty_values": penalties}, file)


if __name__ == "__main__":
    benchmark()
