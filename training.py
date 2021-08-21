from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import util
import random
import sys


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
# train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0]]
misinfo_data = util.json_preprocessing("train.json")
not_misinfo_data = util.json_preprocessing("train_not_misinfo.json")
train_data = misinfo_data + not_misinfo_data
# random.shuffle(train_data)

train_df = pd.DataFrame(train_data)
train_df[0] = train_df[0].str.decode("utf-8")


# eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0]]
dev_data = util.json_preprocessing("dev.json")
dev_df = pd.DataFrame(dev_data)
dev_df[0] = dev_df[0].str.decode("utf-8")

# print(dev_df)

args = {
    # "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "fp16": False,
    # "max_seq_length": 10,
    # "train_batch_size": 1,
    # "eval_batch_size": 1,
    "num_train_epochs": 3,
    # "save_eval_checkpoints": False,
    # "save_model_every_epoch": False,
    "evaluate_during_training": True,
    # "evaluate_generated_text": True,
    "evaluate_during_training_verbose": True,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,
    "use_multiprocessing": False,
}

# bert -> bert-base-cased
# roberta -> roberta-base
# albert -> albert-base-v2

# Create a Binary ClassificationModel
if sys.argv[1] == "0":
    print("Train from the start")
    model = ClassificationModel("albert", "albert-base-v2", use_cuda=True, args=args)
else:
    print("Continue training")
    model = ClassificationModel("albert", "outputs/", use_cuda=True, args=args)


# Train the model
model.train_model(train_df, eval_df=dev_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(dev_df)
