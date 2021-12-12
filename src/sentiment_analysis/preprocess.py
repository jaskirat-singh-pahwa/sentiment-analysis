from typing import Union, List, Tuple
import pandas as pd
import glob


'''
Datasets used:
-- Stanford movie reviews dataset: https://ai.stanford.edu/~amaas/data/sentiment/

Sentiment Label:
0 -> Negative
1 -> Positive

'''


def load_data(path: str) -> pd.DataFrame:
    files = glob.glob(path + "/*.csv")

    reviews = pd.DataFrame(columns=["review", "sentiment"])

    for file in files:
        data = pd.read_csv(file, index_col=None, header=0)
        reviews = reviews.append(data)

    return reviews


def separate_positive_and_negative_reviews(dataset):
    positive = dataset.loc[dataset["sentiment"] == "positive"]
    negative = dataset.loc[dataset["sentiment"] == "negative"]

    positive.to_csv("/Users/jaskirat/Illinois/cs-410/TISProject/sentiment-analysis/src/topic_modelling/data/positive"
                    ".csv", index=False)
    negative.to_csv("/Users/jaskirat/Illinois/cs-410/TISProject/sentiment-analysis/src/topic_modelling/data/negative"
                    ".csv", index=False)


def change_label_to_numeric(sentiment: Union[str, int]) -> str:
    if not isinstance(sentiment, int):
        if sentiment.lower() == "positive":
            return "pos"
        elif sentiment.lower() == "negative":
            return "neg"
        else:
            pass
    else:
        return str(sentiment)


def get_review_and_sentiment_as_tuple(data: pd.DataFrame) -> List[Tuple[str, str]]:
    reviews_with_sentiment: List[Tuple[str, str]] = []

    for review, sentiment in data.itertuples(index=False):
        reviews_with_sentiment.append((sentiment, review))

    return reviews_with_sentiment


def remove_special_chars_from_words(review: str) -> List[str]:
    reviews_without_special_chars = []
    for token in review.split(" "):
        remove_beg = False
        remove_end = False
        if token[0] in {"(", '"', "'"}:
            remove_beg = True
        if token[-1] in {".", ",", ";", ":", "?", "!", '"', "'", ")"}:
            remove_end = True

        if remove_beg and remove_end:
            reviews_without_special_chars += [token[0], token[1:-1], token[-1]]
        elif remove_beg:
            reviews_without_special_chars += [token[0], token[1:]]
        elif remove_end:
            reviews_without_special_chars += [token[:-1], token[-1]]
        else:
            reviews_without_special_chars += [token]

    return reviews_without_special_chars


def get_processed_data(data_path: str):
    raw_data = load_data(path=data_path)
    # separate_positive_and_negative_reviews(dataset=raw_data)
    raw_data["sentiment"] = raw_data["sentiment"].apply(change_label_to_numeric)
    review_with_sentiment = get_review_and_sentiment_as_tuple(raw_data)
    cleaned_reviews = [
        (review[0], remove_special_chars_from_words(review[1]))
        for review in review_with_sentiment
    ]

    return cleaned_reviews
