from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import os
import numpy as np


model1=f"models/gpt2/gpt2_test_config_3_dataset2/final_version"
tokenizer1 = AutoTokenizer.from_pretrained(model1)

model2="openai-community/gpt2"
tokenizer2 = AutoTokenizer.from_pretrained(model2)


def reward_len(completions, **kwargs):
    if type(completions) == list:
        lst = []

        for completion in completions:
            reward = 0
            splitted = completion.split(" ")
            for phrase in splitted:
                reward -= abs(4 - len(phrase))

            lst.append(reward / len(splitted))
        return lst
    elif type(completions) == str:
        reward = 0
        splitted = completions.split(" ")
        for phrase in splitted:
            reward -= abs(4 - len(phrase))
        return reward / len(splitted)


def alternative_reward(completions, **kwargs):
    """
    Bonusing long completions
    """
    if type(completions) == list:
        lst = []

        for completion in completions:
            splitted = completion.split(" ")
            lst.append(len(splitted))

        return lst
    elif type(completions) == str:

        splitted = completions.split(" ")
        return len(splitted)


pipeline1 = pipeline(task="text-generation", model=model1, device_map="mps")
answer1 = pipeline1("Fill that sentence")

print(f"Answer form finetuned gpt2: \n{answer1}")
print("Reward from finetuned gpt2:")
print(reward_len(answer1[0]["generated_text"]))

pipeline2 = pipeline(task="text-generation", model=model2, device_map="mps")
answer2 = pipeline2("Fill that sentence")

print(print(f"Answer from basic gpt2: \n{answer2}"))
print("Reward from basic gpt2:")
print(reward_len(answer2[0]["generated_text"]))

reward_1= []

reward_2= []


# for _ in tqdm(range(1000)):
#     answer1 = pipeline1("Tomorrow I'm gonna learn for exam")
#     reward_1.append(alternative_reward(answer1[0]["generated_text"]))
#
#     answer2 = pipeline2("Tomorrow I'm gonna learn for exam")
#
#     reward_2.append(alternative_reward(answer2[0]["generated_text"]))
#
#
#
# print(f"Average for finetuned gpt2: {np.mean(reward_1)}")
# print(f"std for finetuned gpt2: {np.std(reward_1)}")
# print(f"Median for finetuned gpt2: {np.median(reward_1)}\n\n")
#
# print(f"Average for basic gpt2: {np.mean(reward_2)}")
# print(f"std for basic gpt2: {np.std(reward_2)}")
# print(f"Median for basic gpt2: {np.median(reward_2)}")