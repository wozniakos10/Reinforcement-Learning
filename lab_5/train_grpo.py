from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import json
PATH_TO_LOGS= "logs/gpt2/config_1/"
PATH_TO_SAVE="models/gpt2/gpt2_test_config_1_dataset2"

dataset = load_dataset('data/bigger_dataset/', data_files='bigger_dataset.txt', split='train')

dataset = dataset.rename_column('text', 'prompt')


# dataset = dataset.map(lambda x: {'completion': ''})

# Define the reward function, which rewards completions that are close to 4 characters
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


# Configure the training arguments
training_args = GRPOConfig(
    output_dir=PATH_TO_SAVE,
    logging_steps=10,
    report_to="tensorboard",
    beta=0,

    # per_device_train_batch_size=4,  # Small batch size for the small dataset
    # gradient_accumulation_steps=2,  # Accumulate gradients
    num_train_epochs=8,  # A few epochs should be enough for this small dataset
**{"logging_dir": PATH_TO_LOGS},
)
json_format = training_args.to_dict()

with open(f"{PATH_TO_SAVE}/config.json", "w") as f:
    f.write(json.dumps(json_format))

# Create the trainer
trainer = GRPOTrainer(
    model="openai-community/gpt2",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,

)

# Train the model
trainer.train()

trainer.save_model(f"{PATH_TO_SAVE}/final_version")