import json
import itertools
import pandas as pd

# Read the data tweets from csv(s) and put in a list, the label is 0 because they are not climate change misinformation tweet
def not_misinformation_preprocessing():
    not_misinfo_data = []

    baroba = pd.read_csv("raw-data/BarackObama.csv", encoding="ISO-8859-1")
    txt_emo = pd.read_csv("raw-data/text_emotion.csv", encoding="ISO-8859-1")
    apple = pd.read_csv(
        "raw-data/Apple-Twitter-Sentiment-DFE.csv", encoding="ISO-8859-1"
    )
    women = pd.read_csv("raw-data/womenmarch.csv", encoding="ISO-8859-1")
    nuclear = pd.read_csv("raw-data/sentiment_nuclear_power.csv", encoding="ISO-8859-1")
    progressive = pd.read_csv(
        "raw-data/progressive-tweet-sentiment.csv", encoding="ISO-8859-1"
    )
    administra = pd.read_csv(
        "raw-data/administration_tweets.csv", encoding="ISO-8859-1"
    )
    newsarticle = pd.read_csv("raw-data/newsarticles_tweets.csv", encoding="ISO-8859-1")
    disaster = pd.read_csv(
        "raw-data/socialmedia-disaster-tweets-DFE.csv", encoding="ISO-8859-1"
    )

    for i in itertools.chain(
        baroba["text"][:250],
        txt_emo["content"][:500],
        apple["text"][:250],
        women["text"][:250],
        nuclear["tweet"][:250],
        progressive["tweet"][:250],
        administra["text"][:250],
        newsarticle["tweets"][:250],
        disaster["text"][:250],
    ):
        cur_list = []
        cur_list.append(i.replace("#", ""))
        cur_list.append("0")
        not_misinfo_data.append(cur_list)

    return not_misinfo_data


# Put that data list in json format and write to file
if __name__ == "__main__":
    not_misinfo_data = not_misinformation_preprocessing()
    data = {}
    k = 0

    for i in not_misinfo_data:
        data["train-" + str(k)] = {"text": i[0], "label": i[1]}
        k += 1

    with open("train_not_misivfnfo.json", "w+") as outF:
        json.dump(data, outF)
