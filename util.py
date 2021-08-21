import json

# Read the training file in json
def json_preprocessing(filename):
    train = open(filename, "r")
    values = json.load(train)

    training_data = []

    for i in values:
        cur_list = []
        cur_list.append(
            values[i]["text"]
            .replace("\n", " ")
            .replace("#", "")
            .replace("\xa0", "")
            .encode("ascii", errors="ignore")
        )
        cur_list.append(int(values[i]["label"]))
        training_data.append(cur_list)

    train.close()
    return training_data


# Read the test-unlabelled.json
def load_json(filename):
    data = open(filename, "r")
    values = json.load(data)

    result = []
    for i in values:
        result.append(
            values[i]["text"]
            .replace("\n", " ")
            .replace("#", "")
            .replace("\xa0", "")
            .encode("ascii", errors="ignore")
        )
    result.close()
    return result