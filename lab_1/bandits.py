import matplotlib.pyplot as plt
import numpy as np
from copy import copy

from abc import abstractmethod
from itertools import accumulate
import random
from typing import Protocol

from functools import wraps



class KArmedBandit(Protocol):
    @abstractmethod
    def arms(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def reward(self, arm: str) -> float:
        raise NotImplementedError


class BanditLearner(Protocol):
    name: str
    color: str
    # Hardcoded stationarity for learners.
    is_stationary: bool = False

    @abstractmethod
    def reset(self, arms: list[str], time_steps: int):
        raise NotImplementedError

    @abstractmethod
    def pick_arm(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def acknowledge_reward(self, arm: str, reward: float) -> None:
        pass


class BanditProblem:
    def __init__(self, time_steps: int, bandit: KArmedBandit, learner: BanditLearner):
        self.time_steps: int = time_steps
        self.bandit: KArmedBandit = bandit
        self.learner: BanditLearner = learner
        self.learner.reset(self.bandit.arms(), self.time_steps)


    def run(self) -> list[float]:
        rewards = []
        # lst to calculate_regret, useful for non-stationary where we need to track best action per step
        max_val_per_step = []

        for _ in range(self.time_steps):
            arm = self.learner.pick_arm()
            reward = self.bandit.reward(arm)
            self.learner.acknowledge_reward(arm, reward)
            rewards.append(reward)
            max_val_per_step.append(max(self.bandit.potential_hits.values()))

            # Checking if stationary or not
            if not self.learner.is_stationary:
                # adjusting values to be in range(0,1)
                for key,values in self.bandit.potential_hits.items():
                    new_value = self.bandit.potential_hits[key] + np.random.normal() / 10
                    self.bandit.potential_hits[key] = np.clip(new_value, 0, 1)
        return rewards, max_val_per_step


POTENTIAL_HITS = {
    "In Praise of Dreams": 0.8,
    "We Built This City": 0.9,
    "April Showers": 0.5,
    "Twenty Four Hours": 0.3,
    "Dirge for November": 0.1,
}


class TopHitBandit(KArmedBandit):
    def __init__(self, potential_hits: dict[str, float]):
        self.potential_hits: dict[str, float] = potential_hits

    def arms(self) -> list[str]:
        return list(self.potential_hits)

    def reward(self, arm: str) -> float:
        thumb_up_probability = self.potential_hits[arm]
        return 1.0 if random.random() <= thumb_up_probability else 0.0


class RandomLearner(BanditLearner):
    def __init__(self):
        self.name = "Random"
        self.color = "black"
        self.arms: list[str] = []

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms

    def pick_arm(self) -> str:
        return random.choice(self.arms)

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        pass


class ExploreThenCommitLearner(BanditLearner):
    def __init__(self, m, color):
        self.m = m
        self.name = f"Explore-Then-Commit - m = {m}"
        self.color = color
        self.arms: list[str] = []
        self.step = 1
        self.total_exploration_steps = None
        self.Q = None
        self.N = None
        self.current_action = None

    @property
    def k(self):
        return len(self.arms)

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.Q = np.zeros((self.k, 1))
        self.N = np.zeros((self.k, 1))
        self.total_exploration_steps = self.k *  self.m
        self.step=0


    def pick_arm(self):
        if self.step <= self.total_exploration_steps:
            self.current_action = self.step % self.k
            return self.arms[self.current_action]
        else:
            self.current_action = np.argmax(self.Q)
            return self.arms[self.current_action]


    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.N[self.current_action] += 1
        self.Q[self.current_action] += (reward - self.Q[self.current_action]) / self.N[self.current_action]
        self.step += 1



class GreedyLearner(BanditLearner):
    def __init__(self, eps, color, name, optimistic_value):
        self.name = name
        self.color = color
        self.arms: list[str] = []
        self.optimistic_value = optimistic_value
        self.Q = None
        self.N = None
        self.eps = eps
        self.current_action = None
        self.step = 0
        self.learning_rate = 0.1

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.step = 0
        self.N = np.zeros((len(arms), 1))
        self.Q = np.zeros((len(arms), 1)) + self.optimistic_value

    def pick_arm(self) -> str:
        # exploration phase
        if self.step < len(self.arms):
            self.current_action = self.step
            return self.arms[self.current_action]
        prob = np.random.uniform(0,1)

        if prob <= self.eps:
            self.current_action = random.randint(0,len(self.arms) - 1)
            return self.arms[self.current_action]
        else:
            self.current_action = np.argmax(self.Q)
            return self.arms[self.current_action]

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.N[self.current_action] += 1
        if self.is_stationary:
            self.Q[self.current_action] += (reward - self.Q[self.current_action]) / self.N[self.current_action]
        else:
            self.Q[self.current_action] += (reward - self.Q[self.current_action]) * self.learning_rate

        self.step+=1

class UpperConfidenceBoundLearner(BanditLearner):
    def __init__(self, name, color, c):
        self.name =  name
        self.color = color
        self.arms: list[str]
        self.Q = None
        self.N = None
        self.c = c
        self.step = 0
        self.current_action = None


    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.step=0
        self.N = np.zeros((len(arms), 1))
        self.Q = np.zeros((len(arms), 1))

    def pick_arm(self) -> str:

        # exploration phase
        if self.step < len(self.arms):
            self.current_action = self.step
            return self.arms[self.current_action]

        self.current_action = np.argmax(self.Q + self.c * np.sqrt(np.log(self.step) / self.N))

        return self.arms[self.current_action]

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.step+=1
        self.N[self.current_action] += 1

        self.Q[self.current_action] += (reward - self.Q[self.current_action]) / self.N[self.current_action]

class GradientLearner(BanditLearner):
    def __init__(self, name,alfa, color, use_baseline = True):
        self.name = name
        self.color = color
        self.arms: list[str] = []
        # Bigger H, bigger chance to select particular arm
        self.H = None
        self.Q = 0
        self.N = 0
        self.alfa = alfa

        self.step = 0
        self.use_baseline = use_baseline
        self.current_action = None
        self.current_action_negation = None

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.H = np.zeros((len(arms), 1))
        self.Q = 0
        self.N = 0
        self.step = 0


    def pick_arm(self) -> str:
        # exploration phase
        if self.step < len(self.arms):
            self.current_action = self.step
            return self.arms[self.current_action]

        prob_distribution = self.get_all_arms_probabilities().flatten()
        # Choosing based on current distribution
        self.current_action = np.random.choice(len(self.arms), p=prob_distribution)
        self.current_action_negation = [elem for elem in range(len(self.arms)) if elem != self.current_action]
        return self.arms[self.current_action]

    def acknowledge_reward(self, arm: str, reward:  float) -> None:
        self.step+=1
        self.H[self.current_action]+= self.alfa * (reward - self.Q) * (1 - self.softmax_probability())
        self.H[self.current_action_negation]-= self.alfa * (reward - self.Q) * self.softmax_probability(is_current_action=False)

        self.N+= 1
        if self.use_baseline:
            self.Q+= (reward - self.Q) / self.N

    def softmax_probability(self, is_current_action=True):
        if is_current_action:
            return (np.exp(self.H[self.current_action])) / (np.sum(np.exp(self.H)))
        else:
            return (np.exp(self.H[self.current_action_negation])) / (np.sum(np.exp(self.H)))

    def get_all_arms_probabilities(self):
        return np.exp(self.H) / np.sum(np.exp(self.H))

class ThompsonSampling(BanditLearner):
    def __init__(self):
        self.name = "Thompson Sampling Learner"
        self.color = "pink"
        self.arms: list[str]
        self.alfa: np.ndarray = np.ones((5, 1))
        self.beta: np.ndarray = np.ones((5, 1))
        self.step = 0
        self.current_action = None

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.alfa = np.ones((5, 1))
        self.beta = np.ones((5, 1))
        self.step = 0

    def pick_arm(self) -> str:

        if self.step < len(self.arms):
            self.current_action = self.step
            return self.arms[self.current_action]

        samples = [np.random.beta(self.alfa[elem], self.beta[elem]) for elem in range(len(self.arms))]
        self.current_action = np.argmax(samples)

        return self.arms[self.current_action]

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.step += 1
        self.alfa[self.current_action] += reward
        self.beta[self.current_action] += 1 - reward




TIME_STEPS = 1000
TRIALS_PER_LEARNER = 50


def evaluate_learner(learner: BanditLearner) -> None:
    runs_results = []
    for _ in range(TRIALS_PER_LEARNER):
        bandit = TopHitBandit(copy(POTENTIAL_HITS))
        problem = BanditProblem(time_steps=TIME_STEPS, bandit=bandit, learner=learner)
        rewards = problem.run()
        accumulated_rewards = list(accumulate(rewards))
        runs_results.append(accumulated_rewards)

    runs_results, _ = np.array(runs_results)
    mean_accumulated_rewards = np.mean(runs_results, axis=0)
    std_accumulated_rewards = np.std(runs_results, axis=0)
    plt.plot(mean_accumulated_rewards, label=learner.name, color=learner.color)
    plt.fill_between(
        range(len(mean_accumulated_rewards)),
        mean_accumulated_rewards - std_accumulated_rewards,
        mean_accumulated_rewards + std_accumulated_rewards,
        color=learner.color,
        alpha=0.1,
    )



def main():
    learners = [ThompsonSampling(), ExploreThenCommitLearner(m=5, color="violet"), ExploreThenCommitLearner(m=10, color="red"), ExploreThenCommitLearner(m=20, color="green"), ExploreThenCommitLearner(m=50, color="yellow"),RandomLearner()]
    for learner in learners:
        evaluate_learner(learner)

    plt.xlabel('Time')
    plt.ylabel('Sum of the rewards')
    plt.title("Sum of the rewards in time")
    plt.ylim(0, TIME_STEPS)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
