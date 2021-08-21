from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import util
import json

args = {
    # "reprocess_input_data": True,
    # "overwrite_output_dir": True,
    # "fp16": False,
    # "max_seq_length": 10,
    # "train_batch_size": 1,
    # "eval_batch_size": 1,
    # "num_train_epochs": 2,
    # "save_eval_checkpoints": False,
    # "save_model_every_epoch": False,
    # "evaluate_during_training": True,
    # "evaluate_generated_text": True,
    # "evaluate_during_training_verbose": True,
    "use_multiprocessing": False
}

# Load the model in /outputs
model = ClassificationModel("albert", "outputs/", use_cuda=True, args=args)

test_text = util.load_json("data/test-unlabelled.json")
test_text = [x.decode("utf-8") for x in test_text]

dev_text = util.load_json("data/dev.json")
dev_text = [x.decode("utf-8") for x in dev_text]

# predictions, raw_outputs = model.predict(['why climate change seems to have faded from the news in usa Francis Menton, in the US journal \Manhattan Contrarian\' explains why climate change seems to have faded by showing data from the easily-available UAH global lower troposphere record, derived from satellite sensors.  That record exists from 1979 to present, shown in the latest chart from UAH going through the end of June 2018.', "Ahjhj"])
# Predict the label for the test-unlablled, and write to the file test-output.json
predictions, raw_outputs = model.predict(test_text)

predict_output = {}
k = 0
for i in predictions:
    predict_output["test-" + str(k)] = {"label": int(i)}
    k += 1

with open("data/test-output.json", "w+") as outF:
    json.dump(predict_output, outF)


# Predict the label for the dev (only using the text), and write to the file dev-output.json
dev_predictions, dev_raw_outputs = model.predict(dev_text)

dev_output = {}
k = 0
for i in dev_predictions:
    dev_output["dev-" + str(k)] = {"label": int(i)}
    k += 1

with open("dev-output.json", "w+") as outDev:
    json.dump(dev_output, outDev)
