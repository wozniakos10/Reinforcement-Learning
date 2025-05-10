import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Literal, Type
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
from sympy.physics.units import action
from models import ActorCriticCommonBeginning, Actor, Critic
from logger import configure_logger

# -------------- PARAMETERS SETTING -------------------
torch.manual_seed(0)
N = 5000
IS_COMMON = False
IS_TRUNCATED = True
H1_SIZE = 1024
H2_SIZE = 256
LR = 0.00001
DISCOUNT_FACTOR = 0.99
SAVE_PREFIX = "landing/detached_actor_critic_1024_256_lr_00001"
logger_instance = configure_logger(f"{SAVE_PREFIX}/learning_logger.log")
PROBLEM: Literal["CartPole-v1", "LunarLander-v3"] = "LunarLander-v3"

# -------------- PARAMETERS SETTING -------------------


# ---------- DEVICE CHOSING ---------------------
# Sprawdzenie i wybór najlepszego dostępnego urządzenia
if torch.backends.mps.is_available():
    # Używanie Metal Performance Shaders na macOS
    device = torch.device("mps")
    print("Używanie urządzenia MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    # Używanie CUDA, jeśli dostępne (mało prawdopodobne na macOS)
    device = torch.device("cuda")
    print("Używanie urządzenia CUDA")
else:
    # Używanie CPU jako ostateczność
    device = torch.device("cpu")
    print("Używanie CPU")


# ---------- DEVICE CHOSING ---------------------


class ActorCriticController:
    def __init__(self, environment, learning_rate: float, discount_factor: float, is_common: bool = True, is_evaluation: bool = False) -> None:
        self.environment = environment
        self.state_shape = environment.observation_space.shape
        self.action_size = int(environment.action_space.n)
        self.discount_factor: float = discount_factor
        self.is_common: bool = is_common
        self.is_evaluation: bool = is_evaluation
        self.common_model: nn.Module = self.create_model(
            ActorCriticCommonBeginning,
            (self.state_shape[0], H1_SIZE, H2_SIZE, self.action_size),
            is_evaluation=is_evaluation, path = f"{SAVE_PREFIX}/common_model.pt"
        ).to(device)

        self.actor_model: nn.Module = self.create_model(
            Actor,
            (self.state_shape[0], H1_SIZE, H2_SIZE, self.action_size),
            is_evaluation=is_evaluation, path = f"{SAVE_PREFIX}/actor_final_model.pt"
        ).to(device)

        self.critic_model: nn.Module = self.create_model(
            Critic,
            (self.state_shape[0], H1_SIZE, H2_SIZE),
            is_evaluation=is_evaluation, path = f"{SAVE_PREFIX}/critic_final_model.pt"
        ).to(device)

        self.common_optimizer: Optional[torch.optim.Optimizer] = optim.Adam(
            self.common_model.parameters(), lr=learning_rate
        )

        self.actor_optimizer: Optional[torch.optim.Optimizer] = optim.Adam(
            self.actor_model.parameters(), lr=learning_rate
        )

        self.critic_optimizer: Optional[torch.optim.Optimizer] = optim.Adam(
            self.critic_model.parameters(), lr=learning_rate
        )

        self.saved_log_probs: Optional[torch.Tensor] = None
        self.last_error_squared: float = 0.0


    @staticmethod
    def create_model(model_class: Type[nn.Module], model_args: tuple, is_evaluation: bool, path: str = "") -> nn.Module:
        model = model_class(*model_args)
        if is_evaluation:
            try:
                model.load_state_dict(torch.load(path))
                model.eval()
            except FileNotFoundError:
                print(f"Cannot load model weights from {path}")

        return model

    def choose_action(self, state: np.ndarray) -> int:
        state = self.format_state(state)
        state_tensor = torch.FloatTensor(state).to(device)

        if self.is_common:
            self.common_optimizer.zero_grad()
            action_probs = self.common_model(state_tensor, type="actor")
        else:
            self.actor_optimizer.zero_grad()
            action_probs = self.actor_model(state_tensor)

        action_distribution = torch.distributions.Categorical(action_probs)

        if self.is_evaluation:
            # W trybie ewaluacji wybieramy akcję deterministycznie (maksymalne prawdopodobieństwo)
            action = torch.argmax(action_probs, dim=-1)
            self.saved_log_probs = None  # Nie zapisujemy log_prob w ewaluacji
        else:
            # W trybie treningowym losujemy z rozkładu
            action = action_distribution.sample()
            self.saved_log_probs = action_distribution.log_prob(action)

        return int(action.cpu().item())

    def learn(self, state: np.ndarray, reward: float, new_state: np.ndarray, terminal: bool) -> None:
        state = self.format_state(state)
        new_state = self.format_state(new_state)

        # Konwersja do tensorów PyTorch i przeniesienie na urządzenie
        state_tensor = torch.FloatTensor(state).to(device)
        new_state_tensor = torch.FloatTensor(new_state).to(device)
        reward_tensor = torch.FloatTensor([reward]).to(device)

        if self.is_common:
            # Wartość bieżącego stanu
            current_value = self.common_model(state_tensor, type="critic")

        else:
            self.critic_optimizer.zero_grad()
            current_value = self.critic_model(state_tensor)

        # Obliczenie TD error
        if terminal:
            # W stanie terminalnym nie ma następnej wartości
            target = reward_tensor
        else:
            # Używamy detach() aby zatrzymać przepływ gradientu
            if self.is_common:
                next_value = self.common_model(new_state_tensor, type="critic").detach()
            else:
                next_value = self.critic_model(new_state_tensor).detach()

            target = reward_tensor + self.discount_factor * next_value

        # TD error (delta)
        delta = target - current_value

        # Zapisanie kwadratu błędu do późniejszego użycia
        self.last_error_squared = delta.cpu().item() ** 2  # Przeniesienie na CPU do zapisu

        # Funkcja straty krytyka (MSE)
        critic_loss = delta ** 2

        # Funkcja straty aktora (policy gradient)
        # Sprawdzenie, czy saved_log_probs jest na odpowiednim urządzeniu
        if self.saved_log_probs.device != device:
            self.saved_log_probs = self.saved_log_probs.to(device)

        # Używamy detach(), ponieważ chcemy, aby delta wpływała na actor loss jako skalar
        actor_loss = -delta.detach() * self.saved_log_probs

        # Całkowita strata (połączenie aktora i krytyka)
        total_loss = critic_loss + actor_loss

        if self.is_common:
            # Propagacja wsteczna
            total_loss.backward()
            # Aktualizacja parametrów
            self.common_optimizer.step()

        else:
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def value_state(self, state: np.ndarray) -> np.ndarray:
        state = self.format_state(state)
        state_tensor = torch.FloatTensor(state).to(device)
        if self.is_common:
            return self.common_model(state_tensor, type="critic")

        else:
            return self.critic_model(state_tensor)

    @staticmethod
    def format_state(state: np.ndarray) -> np.ndarray:
        return np.reshape(state, (1, state.size))


def train_loop() -> None:
    environment = gym.make(PROBLEM)  # zamień na gym.make('LunarLander-v2', render_mode='human') by zająć się lądownikiem
    # zmień lub usuń render_mode, by przeprowadzić trening bez wizualizacji środowiska
    controller = ActorCriticController(environment, LR, DISCOUNT_FACTOR, is_common=IS_COMMON)
    past_rewards = []
    past_errors = []
    for i_episode in tqdm(range(N)):  # tu decydujemy o liczbie epizodów
        done = False
        truncated = False
        state, info = environment.reset()
        reward_sum = 0.0
        errors_history = []

        while not done:
            # environment.render()  # tą linijkę możemy wykomentować, jeżeli nie chcemy mieć wizualizacji na żywo

            action = controller.choose_action(state)
            new_state, reward, done, truncated, info = environment.step(action)
            controller.learn(state, reward, new_state, done)
            state = new_state
            reward_sum += reward
            errors_history.append(controller.last_error_squared)

            if IS_TRUNCATED:
                if truncated:
                    break

        past_rewards.append(reward_sum)
        past_errors.append(np.mean(errors_history))

        window_size = 50  # tutaj o rozmiarze okienka od średniej kroczącej
        if i_episode % 200 == 0 or i_episode == N - 1:  # tutaj o częstotliwości zrzucania wykresów
            reward_means = [0]
            if len(past_rewards) >= window_size:
                fig, axs = plt.subplots(2)

                # Obliczamy średnie ruchome dla wykresów (zachowując oryginalne wywołania)
                error_means = [np.mean(past_errors[i:i + window_size]) for i in range(len(past_errors) - window_size)]
                reward_means = [np.mean(past_rewards[i:i + window_size]) for i in
                                range(len(past_rewards) - window_size)]

                # Wykres błędów
                axs[0].plot(error_means, 'tab:red')
                axs[0].set_title('mean squared error')

                # Ustawiamy skalę osi Y dla wykresu błędów
                min_error = min(error_means)
                if min_error > 0:
                    axs[0].set_ylim(bottom=0)  # Jeśli min > 0, zaczynamy od 0
                else:
                    axs[0].set_ylim(bottom=min_error)  # W przeciwnym razie od wartości minimalnej

                # Wykres nagród
                axs[1].plot(reward_means, 'tab:green')
                axs[1].set_title('sum of rewards')

                # Ustawiamy skalę osi Y dla wykresu nagród
                min_reward = min(reward_means)
                if min_reward > 0:
                    axs[1].set_ylim(bottom=0)  # Jeśli min > 0, zaczynamy od 0
                else:
                    axs[1].set_ylim(bottom=min_reward)

            plt.tight_layout()
            plt.savefig(f'{SAVE_PREFIX}/plots/learning_{i_episode}.png')
            plt.clf()
            logger_instance.info(f"Saved plot to {SAVE_PREFIX}/plots/learning_{i_episode}.png")
            logger_instance.info(f"Last moving average reward: {reward_means[-1]}")

        if i_episode % 100 == 0:
            logger_instance.info(f"Sucessfuly processed {i_episode} / {N} episodes.")

        if i_episode % 200 == 0:
            if controller.is_common:
                torch.save(controller.common_model.state_dict(),
                           f'{SAVE_PREFIX}/common_episode_{i_episode}_model.pt')

            else:
                torch.save(controller.actor_model.state_dict(),
                           f'{SAVE_PREFIX}/actor_episode_{i_episode}_model.pt')
                torch.save(controller.critic_model.state_dict(),
                           f'{SAVE_PREFIX}/critic_episode_{i_episode}_model.pt')


    environment.close()

    logger_instance.info("Learning process is finished!")

    if controller.is_common:
        torch.save(controller.common_model.state_dict(),
                   f'{SAVE_PREFIX}/common_final_model.pt')

    else:
        torch.save(controller.actor_model.state_dict(),
                   f'{SAVE_PREFIX}/actor_final_model.pt')
        torch.save(controller.critic_model.state_dict(),
                   f'{SAVE_PREFIX}/critic_final_model.pt')


def evaluation_loop() -> None:
    environment = gym.make(PROBLEM, render_mode="human")  # zamień na gym.make('LunarLander-v2', render_mode='human') by zająć się lądownikiem
    # zmień lub usuń render_mode, by przeprowadzić trening bez wizualizacji środowiska
    controller = ActorCriticController(environment, LR, DISCOUNT_FACTOR, is_common=IS_COMMON, is_evaluation=True)
    for _ in tqdm(range(10)):  # tu decydujemy o liczbie epizodów
        done = False
        state, info = environment.reset()


        while not done:
            environment.render()  # tą linijkę możemy wykomentować, jeżeli nie chcemy mieć wizualizacji na żywo
            action = controller.choose_action(state)
            new_state, reward, done, truncated, info = environment.step(action)
            state = new_state


            if IS_TRUNCATED:
                if truncated:
                    break


    environment.close()


def test_state_valuating() -> None:
    environment = gym.make(PROBLEM)
    controller = ActorCriticController(environment, LR, DISCOUNT_FACTOR, is_common=IS_COMMON, is_evaluation=True)
    if PROBLEM == "CartPole-v1":
        vertical_pool_zero_velocity_state = np.array([0,0,0,0])
        v = controller.value_state(vertical_pool_zero_velocity_state)
        print(f"Wartosciowanie stanu dla pionowego kijka i zerowych predkosci: {v}")

        pool_start_fast_downfall = np.array([0, 0, 0.1, 5])
        v = controller.value_state(pool_start_fast_downfall)
        print(f"Wartosciowanie stanu dla kijka zaczynajacego spadac: {v}")

        trolley_close_to_border = np.array([4.3, 0, 0, 0])
        v = controller.value_state(trolley_close_to_border)
        print(f"Wartosciowanie stanu dla kijka blisko krawedzi: {v}")
    else:
        close_to_landing_state = np.array([0, 0.01, 0,-0.5,0,0,0,0])
        v = controller.value_state(close_to_landing_state)
        print(f"Wartosciowanie stanu dla chwile przed ladowaniem: {v}")


        turn_to_left_state = np.array([0, 0, 0, 0.5, -6, 0, 0, 0])
        v = controller.value_state(turn_to_left_state)
        print(f"Wartosciowanie stanu dla mocno przechylonegow lewo: {v}")

        action = controller.choose_action(turn_to_left_state)
        print(f"Proponowana akcja dla statku mocno przechylonego w lewo: {action}")

        turn_to_right_state = np.array([0, 0, 0, 0.5, 6, 0, 0, 0])
        v = controller.value_state(turn_to_right_state)
        print(f"Wartosciowanie stanu dla mocno przechylonego w prawo: {v}")

        action = controller.choose_action(turn_to_right_state)
        print(f"Proponowana akcja dla statku mocno przechylonego w prawo: {action}")

if __name__ == '__main__':
    # train_loop()
    # evaluation_loop()
    test_state_valuating()