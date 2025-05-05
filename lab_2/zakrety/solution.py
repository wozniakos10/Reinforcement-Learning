from __future__ import annotations
import collections
import random

import numpy as np
import sklearn.preprocessing as skl_preprocessing

from problem import Action, available_actions, Corner, Driver, Experiment, Environment, State

ALMOST_INFINITE_STEP = 100000
MAX_LEARNING_STEPS = 500


class RandomDriver(Driver):
    def __init__(self):
        self.current_step: int = 0

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        return random.choice(available_actions(state))

    def control(self, state: State, last_reward: int, is_validation=False) -> Action:
        self.current_step += 1
        return random.choice(available_actions(state))

    def finished_learning(self) -> bool:
        return self.current_step > MAX_LEARNING_STEPS


class OffPolicyNStepSarsaDriver(Driver):
    def __init__(self, step_size: float, step_no: int, experiment_rate: float, discount_factor: float) -> None:
        self.step_size: float = step_size
        self.step_no: int = step_no
        self.experiment_rate: float = experiment_rate
        self.discount_factor: float = discount_factor
        self.q: dict[tuple[State, Action], float] = collections.defaultdict(float)
        self.current_step: int = 0
        self.final_step: int = ALMOST_INFINITE_STEP
        self.finished: bool = False
        self.states: dict[int, State] = dict()
        self.actions: dict[int, Action] = dict()
        self.rewards: dict[int, int] = dict()

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        self.states[self._access_index(self.current_step)] = state
        action = self._select_action(self.epsilon_greedy_policy(state, available_actions(state)))
        self.actions[self._access_index(self.current_step)] = action
        self.final_step = ALMOST_INFINITE_STEP
        self.finished = False
        return action

    def control(self, state: State, last_reward: int, is_validation: bool = False) -> Action:
        if is_validation:
            if last_reward == 0:
                return Action(0, 0)
            action = self._select_action(self.greedy_policy(state, available_actions(state)))
            return action

        if self.current_step < self.final_step:
            self.rewards[self._access_index(self.current_step + 1)] = last_reward
            self.states[self._access_index(self.current_step + 1)] = state
            if self.final_step == ALMOST_INFINITE_STEP and (
                last_reward == 0 or self.current_step == MAX_LEARNING_STEPS
            ):
                self.final_step = self.current_step

            action = self._select_action(self.epsilon_greedy_policy(state, available_actions(state)))

            self.actions[self._access_index(self.current_step + 1)] = action
        else:
            action = Action(0, 0)

        update_step = self.current_step - self.step_no + 1
        if update_step >= 0:
            return_value_weight = self._return_value_weight(update_step)
            return_value = self._return_value(update_step)
            state_t = self.states[self._access_index(update_step)]
            action_t = self.actions[self._access_index(update_step)]
            self.q[state_t, action_t] += (
                self.step_size * return_value_weight * (return_value - self.q[state_t, action_t])
            )  # TODO: Tutaj trzeba zaktualizować tablicę wartościującą akcje Q

        if update_step == self.final_step - 1:
            self.finished = True

        self.current_step += 1
        return action

    def _return_value(self, update_step):
        return_value = 0.0
        for i in range(update_step + 1, min(update_step + self.step_no + 1, self.final_step + 1)):
            return_value += self.discount_factor ** (i - update_step - 1) * self.rewards[self._access_index(i)]
        if update_step + self.step_no < self.final_step:
            state = self.states[self._access_index(update_step + self.step_no)]
            action = self.actions[self._access_index(update_step + self.step_no)]
            return_value += self.discount_factor ** (self.step_no) * self.q[state, action]
        # TODO: Tutaj trzeba policzyć zwrot G
        return return_value

    def _return_value_weight(self, update_step):
        return_value_weight = 1.0
        for i in range(update_step + 1, min(update_step + self.step_no + 1, self.final_step)):
            state = self.states[self._access_index(i)]
            action = self.actions[self._access_index(i)]

            # pi policy
            greedy_action = self._select_action(self.greedy_policy(state, available_actions(state)))
            pi_policy_value = 1 if action == greedy_action else 0

            # b policy.
            b_policy_value = 1 - self.experiment_rate if action == greedy_action else 0
            # adding probability to draw particular action
            b_policy_value += self.experiment_rate / len(available_actions(state))

            ratio = pi_policy_value / b_policy_value

            if np.isnan(ratio):
                continue

            return_value_weight *= ratio

        if np.isnan(return_value_weight):
            return 1.0

        return return_value_weight

    def finished_learning(self) -> bool:
        return self.finished

    def _access_index(self, index: int) -> int:
        return index % (self.step_no + 1)

    @staticmethod
    def _select_action(actions_distribution: dict[Action, float]) -> Action:
        actions = list(actions_distribution.keys())
        probabilities = list(actions_distribution.values())
        try:
            i = np.random.choice(list(range(len(actions))), p=probabilities)
        except Exception as e:
            print(f"Caught exception: {e}")
            print(actions_distribution)
            print(probabilities)
            print(sum(probabilities))
            raise Exception
        return actions[i]

    def epsilon_greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        if np.random.random() < self.experiment_rate:
            probabilities = self._random_probabilities(actions)
        else:
            probabilities = self._greedy_probabilities(state, actions)
        # probabilities = None  # TODO: tutaj trzeba ustalic prawdopodobieństwa wyboru akcji według polityki ε-zachłannej
        return {action: probability for action, probability in zip(actions, probabilities)}

    def greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        probabilities = self._greedy_probabilities(state, actions)
        return {action: probability for action, probability in zip(actions, probabilities)}

    def _greedy_probabilities(self, state: State, actions: list[Action]) -> np.ndarray:
        values = np.array([self.q[state, action] for action in actions])
        maximal_spots = (values == np.max(values)).astype(float)

        normalized = self._normalise(maximal_spots)
        if 0 <= sum(normalized) < 0.1:
            print(f"probability do not sum to 1 in greedy!: {normalized}")
            print(state)
            print(actions)
            print(f"Maximal spots: {maximal_spots}")
            print(f"Values: {values}")
        return normalized

    @staticmethod
    def _random_probabilities(actions: list[Action]) -> np.ndarray:
        maximal_spots = np.array([1.0 for _ in actions])
        normalized = OffPolicyNStepSarsaDriver._normalise(maximal_spots)
        if 0 <= sum(normalized) < 0.1:
            print(f"probability do not sum to 1 in random!: {normalized}")
        return normalized

    @staticmethod
    def _normalise(probabilities: np.ndarray) -> np.ndarray:
        return skl_preprocessing.normalize(probabilities.reshape(1, -1), norm="l1")[0]


def main() -> None:
    # experiment = Experiment(
    #     environment=Environment(
    #         corner=Corner(
    #             name='corner_b'
    #         ),
    #         steering_fail_chance=0.01,
    #     ),
    #     driver=RandomDriver(),
    #     number_of_episodes=100,
    # )

    experiment = Experiment(
        environment=Environment(
            corner=Corner(name="corner_d"),
            steering_fail_chance=0.01,
        ),
        driver=OffPolicyNStepSarsaDriver(
            step_no=2,
            step_size=0.8,
            experiment_rate=0.05,
            discount_factor=1.00,
        ),
        number_of_episodes=30000,
    )

    experiment.run()


if __name__ == "__main__":
    main()
