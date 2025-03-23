from bandits import (
    UpperConfidenceBoundLearner,
    GreedyLearner,
    GradientLearner,
    TopHitBandit,
    BanditProblem,
    BanditLearner,
)
import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate
import json

TRIALS_PER_LEARNER = 100
TIME_STEPS = 1000


cfg_lst = [
    {
        "learner": GreedyLearner,
        "label_name": "Ԑ-Greedy",
        "research_parameter": {
            "name": "eps",
            "value": 1 / 128,
        },
        "param": {
            "eps": 1 / 128,
            "color": "red",
            "name": "Greedy_1/128",
            "optimistic_value": 0,
        },
    },
    {
        "learner": GreedyLearner,
        "label_name": "Ԑ-Greedy",
        "research_parameter": {"name": "eps", "value": 1 / 64},
        "param": {
            "eps": 1 / 64,
            "color": "blue",
            "name": "Greedy_1/64",
            "optimistic_value": 0,
        },
    },
    {
        "learner": GreedyLearner,
        "label_name": "Ԑ-Greedy",
        "research_parameter": {"name": "eps", "value": 1 / 32},
        "param": {
            "eps": 1 / 32,
            "color": "green",
            "name": "Greedy_1/32",
            "optimistic_value": 0,
        },
    },
    {
        "learner": GreedyLearner,
        "label_name": "Ԑ-Greedy",
        "research_parameter": {"name": "eps", "value": 1 / 16},
        "param": {
            "eps": 1 / 16,
            "color": "orange",
            "name": "Greedy_1/16",
            "optimistic_value": 0,
        },
    },
    {
        "learner": GreedyLearner,
        "label_name": "Ԑ-Greedy",
        "research_parameter": {"name": "eps", "value": 1 / 8},
        "param": {
            "eps": 1 / 8,
            "color": "purple",
            "name": "Greedy_1/8",
            "optimistic_value": 0,
        },
    },
    {
        "learner": GreedyLearner,
        "label_name": "Ԑ-Greedy",
        "research_parameter": {"name": "eps", "value": 1 / 4},
        "param": {
            "eps": 1 / 4,
            "color": "black",
            "name": "Greedy_1/4",
            "optimistic_value": 0,
        },
    },
    {
        "learner": GreedyLearner,
        "label_name": "Optimistic greedy",
        "research_parameter": {"name": "Q", "value": 1 / 4},
        "param": {
            "eps": 1 / 16,
            "color": "blue",
            "name": "Optimistic Greedy_1/4",
            "optimistic_value": 1 / 4,
        },
    },
    {
        "learner": GreedyLearner,
        "label_name": "Optimistic greedy",
        "research_parameter": {"name": "Q", "value": 1 / 2},
        "param": {
            "eps": 1 / 16,
            "color": "green",
            "name": "Optimistic Greedy_1/2",
            "optimistic_value": 1 / 2,
        },
    },
    {
        "learner": GreedyLearner,
        "label_name": "Optimistic greedy",
        "research_parameter": {"name": "Q", "value": 1},
        "param": {
            "eps": 1 / 16,
            "color": "orange",
            "name": "Optimistic Greedy_1",
            "optimistic_value": 1,
        },
    },
    {
        "learner": GreedyLearner,
        "label_name": "Optimistic greedy",
        "research_parameter": {"name": "Q", "value": 2},
        "param": {
            "eps": 1 / 16,
            "color": "purple",
            "name": "Optimistic Greedy_2",
            "optimistic_value": 2,
        },
    },
    {
        "learner": GreedyLearner,
        "label_name": "Optimistic greedy",
        "research_parameter": {"name": "Q", "value": 4},
        "param": {
            "eps": 1 / 16,
            "color": "black",
            "name": "Optimistic Greedy_4",
            "optimistic_value": 4,
        },
    },
    {
        "learner": UpperConfidenceBoundLearner,
        "label_name": "UCB",
        "research_parameter": {"name": "c", "value": 1 / 16},
        "param": {"c": 1 / 16, "color": "blue", "name": "UCB_1/16"},
    },
    {
        "learner": UpperConfidenceBoundLearner,
        "label_name": "UCB",
        "research_parameter": {"name": "c", "value": 1 / 8},
        "param": {"c": 1 / 8, "color": "green", "name": "UCB_1/8"},
    },
    {
        "learner": UpperConfidenceBoundLearner,
        "label_name": "UCB",
        "research_parameter": {"name": "c", "value": 1 / 4},
        "param": {"c": 1 / 4, "color": "orange", "name": "UCB_1/4"},
    },
    {
        "learner": UpperConfidenceBoundLearner,
        "label_name": "UCB",
        "research_parameter": {"name": "c", "value": 1},
        "param": {"c": 1, "color": "purple", "name": "UCB_1"},
    },
    {
        "learner": UpperConfidenceBoundLearner,
        "label_name": "UCB",
        "research_parameter": {"name": "c", "value": 2},
        "param": {"c": 2, "color": "black", "name": "UCB_2"},
    },
    {
        "learner": UpperConfidenceBoundLearner,
        "label_name": "UCB",
        "research_parameter": {"name": "c", "value": 4},
        "param": {"c": 4, "color": "black", "name": "UCB_4"},
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient with baseline",
        "research_parameter": {"name": "alfa", "value": 1 / 32},
        "param": {
            "alfa": 1 / 32,
            "color": "blue",
            "name": "grad_with_base_1/32",
            "use_baseline": "True",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient with baseline",
        "research_parameter": {"name": "alfa", "value": 1 / 16},
        "param": {
            "alfa": 1 / 16,
            "color": "blue",
            "name": "grad_with_base_1/16",
            "use_baseline": "True",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient with baseline",
        "research_parameter": {"name": "alfa", "value": 1 / 8},
        "param": {
            "alfa": 1 / 8,
            "color": "blue",
            "name": "grad_with_base_1/8",
            "use_baseline": "True",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient with baseline",
        "research_parameter": {"name": "alfa", "value": 1 / 4},
        "param": {
            "alfa": 1 / 4,
            "color": "blue",
            "name": "grad_with_base_1/4",
            "use_baseline": "True",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient with baseline",
        "research_parameter": {"name": "alfa", "value": 1 / 2},
        "param": {
            "alfa": 1 / 2,
            "color": "blue",
            "name": "grad_with_base_1/2",
            "use_baseline": "True",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient with baseline",
        "research_parameter": {"name": "alfa", "value": 1},
        "param": {
            "alfa": 1,
            "color": "blue",
            "name": "grad_with_base_1",
            "use_baseline": "True",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient with baseline",
        "research_parameter": {"name": "alfa", "value": 2},
        "param": {
            "alfa": 2,
            "color": "blue",
            "name": "grad_with_base_2",
            "use_baseline": "True",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient with baseline",
        "research_parameter": {"name": "alfa", "value": 4},
        "param": {
            "alfa": 4,
            "color": "blue",
            "name": "grad_with_base_4",
            "use_baseline": "True",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient without baseline",
        "research_parameter": {"name": "alfa", "value": 1 / 32},
        "param": {
            "alfa": 1 / 32,
            "color": "blue",
            "name": "grad_without_base_1/32",
            "use_baseline": "False",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient without baseline",
        "research_parameter": {"name": "alfa", "value": 1 / 16},
        "param": {
            "alfa": 1 / 16,
            "color": "blue",
            "name": "grad_without_base_1/16",
            "use_baseline": "False",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient without baseline",
        "research_parameter": {"name": "alfa", "value": 1 / 8},
        "param": {
            "alfa": 1 / 8,
            "color": "blue",
            "name": "grad_without_base_1/8",
            "use_baseline": "False",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient without baseline",
        "research_parameter": {"name": "alfa", "value": 1 / 4},
        "param": {
            "alfa": 1 / 4,
            "color": "blue",
            "name": "grad_without_base_1/4",
            "use_baseline": "False",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient without baseline",
        "research_parameter": {"name": "alfa", "value": 1 / 2},
        "param": {
            "alfa": 1 / 2,
            "color": "blue",
            "name": "grad_without_base_1/2",
            "use_baseline": "False",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient without baseline",
        "research_parameter": {"name": "alfa", "value": 1},
        "param": {
            "alfa": 1,
            "color": "blue",
            "name": "grad_without_base_1",
            "use_baseline": "False",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient without baseline",
        "research_parameter": {"name": "alfa", "value": 2},
        "param": {
            "alfa": 2,
            "color": "blue",
            "name": "grad_without_base_2",
            "use_baseline": "False",
        },
    },
    {
        "learner": GradientLearner,
        "label_name": "Gradient without baseline",
        "research_parameter": {"name": "alfa", "value": 4},
        "param": {
            "alfa": 4,
            "color": "blue",
            "name": "grad_without_base_4",
            "use_baseline": "False",
        },
    },
]


def evaluate_learner(learner: BanditLearner) -> tuple[np.ndarray, np.ndarray]:
    POTENTIAL_HITS = {
        "In Praise of Dreams": 0.8,
        "We Built This City": 0.9,
        "April Showers": 0.5,
        "Twenty Four Hours": 0.3,
        "Dirge for November": 0.1,
    }
    accumulated_runs_results = []
    run_results = []
    regret_lst = []
    for _ in range(TRIALS_PER_LEARNER):
        new_hits = {}

        for song in POTENTIAL_HITS.keys():
            # Generating new values for each round
            new_value = np.random.random()
            new_hits[song] = new_value

        bandit = TopHitBandit(new_hits)
        problem = BanditProblem(time_steps=TIME_STEPS, bandit=bandit, learner=learner)
        rewards, max_val_per_step = problem.run()
        accumulated_rewards = list(accumulate(rewards))
        accumulated_runs_results.append(accumulated_rewards)
        run_results += rewards
        regret = (np.sum(max_val_per_step) - np.sum(rewards)) / TIME_STEPS
        regret_lst.append(regret)

    accumulated_runs_results = np.array(accumulated_runs_results)
    mean_accumulated_rewards = np.mean(accumulated_runs_results, axis=0)

    run_results = np.array(run_results)
    mean_rewards_per_step = np.mean(run_results, axis=0)
    mean_regret_per_step = np.mean(regret_lst)

    return mean_accumulated_rewards, mean_rewards_per_step, mean_regret_per_step


def run_benchmark():
    score_result = {}
    for cfg in cfg_lst:
        learner = cfg["learner"](**cfg["param"])
        mean_accumulated_rewards, mean_rewards_per_step, mean_regret_per_step = evaluate_learner(learner)
        if score_result.get(cfg["label_name"]) is None:
            score_result[cfg["label_name"]] = {}
        score_result[cfg["label_name"]][learner.name] = {}
        score_result[cfg["label_name"]][learner.name]["acc_mean_rewards"] = mean_accumulated_rewards
        score_result[cfg["label_name"]][learner.name]["mean_rewards_per_step"] = mean_rewards_per_step
        score_result[cfg["label_name"]][learner.name]["mean_regret_per_step"] = mean_regret_per_step
        score_result[cfg["label_name"]][learner.name]["research_parameter"] = cfg.get("research_parameter")

    return score_result


def plot_result(score_result: dict):
    plt.figure(figsize=(12, 10))

    for key, val in score_result.items():
        items = sorted(
            [(elem["research_parameter"]["value"], elem["mean_regret_per_step"]) for _, elem in val.items()],
            key=lambda x: x[0],
        )
        print(items)
        x = [elem[0] for elem in items]
        y = [elem[1] for elem in items]
        plt.plot(
            x,
            y,
            "o-",
            label=key,
        )

    plt.xscale("log")
    x_axis_values = [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4]
    fractions = ["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"]

    plt.xticks(x_axis_values, fractions)

    plt.grid()
    plt.xlabel("Ԑ, α, c, Q0", fontsize=12)
    plt.ylabel("Average regret over first 1000 steps", fontsize=12)
    plt.title(
        "Average regret comparison - non-stationary version, random reward expected values",
        fontsize=16,
    )
    plt.legend()
    plt.show()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if __name__ == "__main__":
    # score_result = run_benchmark()
    # with open("data/non_stationary_random_reward.json", "w") as f:
    #     f.write(json.dumps(score_result, indent=4, cls=NumpyEncoder))

    with open("data/non_stationary_random_reward.json", "r") as f:
        score_result = json.load(f)

    plot_result(score_result)
