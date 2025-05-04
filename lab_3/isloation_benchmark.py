from isolation import Board, MCTSPlayer, Game, Colour
from tqdm import tqdm
import json
import os
from datetime import datetime
from logger import configure_logger


params_lst = [
    # Konfiguracja 1: UCB vs UCB z różnymi współczynnikami c i  stalym czasem
    {
        "params": {
            "Player_A": {
                "params": {
                    "c_coefficient": 0.5,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            },
            "Player_B": {
                "params": {
                    "c_coefficient": 0.5,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            }
        }
    },

    {
        "params": {
            "Player_A": {
                "params": {
                    "c_coefficient": 0.5,
                    "time_limit": 0.1,
                    "policy": "ucb"
                }
            },
            "Player_B": {
                "params": {
                    "c_coefficient": 0.5,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            }
        }
    },

    {
        "params": {
            "Player_A": {
                "params": {
                    "c_coefficient": 0.5,
                    "time_limit": 0.05,
                    "policy": "ucb"
                }
            },
            "Player_B": {
                "params": {
                    "c_coefficient": 0.5,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            }
        }
    },

    {
        "params": {
            "Player_A": {
                "params": {
                    "c_coefficient": 0.5,
                    "time_limit": 0.3,
                    "policy": "ucb"
                }
            },
            "Player_B": {
                "params": {
                    "c_coefficient": 0.5,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            }
        }
    },

{
        "params": {
            "Player_A": {
                "params": {
                    "c_coefficient": 0.5,
                    "time_limit": 0.5,
                    "policy": "ucb"
                }
            },
            "Player_B": {
                "params": {
                    "c_coefficient": 0.5,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            }
        }
    },

    #  UCB staly czas i rozny c_ocefficient
    {
        "params": {
            "Player_A": {
                "params": {
                    "c_coefficient": 0.3,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            },
            "Player_B": {
                "params": {
                    "c_coefficient": 0.5,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            }
        }
    },

{
        "params": {
            "Player_A": {
                "params": {
                    "c_coefficient": 0.1,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            },
            "Player_B": {
                "params": {
                    "c_coefficient": 0.5,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            }
        }
    },

{
        "params": {
            "Player_A": {
                "params": {
                    "c_coefficient": 1,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            },
            "Player_B": {
                "params": {
                    "c_coefficient": 0.5,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            }
        }
    },

{
        "params": {
            "Player_A": {
                "params": {
                    "c_coefficient": 2,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            },
            "Player_B": {
                "params": {
                    "c_coefficient": 0.5,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            }
        }
    },

   #Epsilon-greedy rozna eksploracja
    {
        "params": {
            "Player_A": {
                "params": {
                    "epsilon": 0.2,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            },
            "Player_B": {
                "params": {
                    "epsilon": 0.05,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            }
        }
    },

{
        "params": {
            "Player_A": {
                "params": {
                    "epsilon": 0.1,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            },
            "Player_B": {
                "params": {
                    "epsilon": 0.05,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            }
        }
    },


{
        "params": {
            "Player_A": {
                "params": {
                    "epsilon": 0.5,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            },
            "Player_B": {
                "params": {
                    "epsilon": 0.05,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            }
        }
    },

{
        "params": {
            "Player_A": {
                "params": {
                    "epsilon": 0.01,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            },
            "Player_B": {
                "params": {
                    "epsilon": 0.05,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            }
        }
    },

{
        "params": {
            "Player_A": {
                "params": {
                    "epsilon": 0.2,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            },
            "Player_B": {
                "params": {
                    "epsilon": 0.2,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            }
        }
    },

    {
        "params": {
            "Player_A": {
                "params": {
                    "epsilon": 0.3,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            },
            "Player_B": {
                "params": {
                    "epsilon": 0.05,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            }
        }
    },


{
        "params": {
            "Player_A": {
                "params": {
                    "epsilon": 0.4,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            },
            "Player_B": {
                "params": {
                    "epsilon": 0.05,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            }
        }
    },


    {
        "params": {
            "Player_A": {
                "params": {
                    "epsilon": 0.5,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            },
            "Player_B": {
                "params": {
                    "epsilon": 0.05,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            }
        }
    },

   {
        "params": {
            "Player_A": {
                "params": {
                    "c_coefficient": 0.1,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            },
            "Player_B": {
                "params": {
                    "epsilon": 0.2,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            }
        }
    },

# best ucb vs epsilon greedy
      {
        "params": {
            "Player_A": {
                "params":  {
                    "epsilon": 0.2,
                    "time_limit": 0.2,
                    "policy": "epsilon_greedy"
                }
            },
            "Player_B": {
                "params": {
                    "c_coefficient": 0.1,
                    "time_limit": 0.2,
                    "policy": "ucb"
                }
            }
        }
    },
]


def run_benchmark(params: list, n=100) -> None:
    os.makedirs("results", exist_ok=True)
    logger = configure_logger("results/benchmark.log")

    for idx, elem in enumerate(tqdm(params, desc="Testing parameter configurations", unit="config")):
        logger.info(f"Testing {idx+1}/{len(params)}")
        red_wins = 0
        blue_wins = 0

        for _ in range(n):
            board = Board(8, 8)
            red_player = MCTSPlayer(**elem["params"]["Player_A"]["params"])
            blue_player = MCTSPlayer(**elem["params"]["Player_B"]["params"])
            game = Game(red_player, blue_player, board)
            game.run(verbose=False)

            if game.winner == Colour.RED:
                red_wins += 1
            else:
                blue_wins += 1

        result = {
            "timestamp": datetime.now().isoformat(),
            "player_A_params": elem["params"]["Player_A"]["params"],
            "player_B_params": elem["params"]["Player_B"]["params"],
            "player_a": red_wins,
            "player_b": blue_wins,
            "total_games": n,
            "player_a_win_rate": red_wins / n,
            "player_b_win_rate": blue_wins / n
        }

        # Tworzenie unikalnej i czytelnej nazwy pliku
        def stringify_params(p):
            return "_".join([f"{k}-{str(v).replace('.', '')}" for k, v in p.items()])

        name_A = stringify_params(elem["params"]["Player_A"]["params"])
        name_B = stringify_params(elem["params"]["Player_B"]["params"])
        filename = f"config_{idx+1}_A_{name_A}_vs_B_{name_B}.json"
        filepath = os.path.join("results", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)

        logger.info(f"Results saved to {filepath}")
        logger.info(f"Sucesfully processed {idx+1}/{len(params)}")






if __name__ == '__main__':
    # Uruchom benchmark i zapisz wyniki
    results = run_benchmark(params_lst, 500)

