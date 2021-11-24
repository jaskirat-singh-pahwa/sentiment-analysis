from typing import Union, List, Tuple
import pandas as pd
import glob


'''
Datasets used:
1.) Stanford movie reviews dataset: https://ai.stanford.edu/~amaas/data/sentiment/
2.) Cornell movie reviews dataset: https://www.cs.cornell.edu/people/pabo/movie-review-data/

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


def get_processed_data(data_path: str) -> List[Tuple[str, List[str]]]:
    raw_data = load_data(path=data_path)
    raw_data["sentiment"] = raw_data["sentiment"].apply(change_label_to_numeric)
    review_with_sentiment = get_review_and_sentiment_as_tuple(raw_data)
    cleaned_reviews = [
        (review[0], remove_special_chars_from_words(review[1]))
        for review in review_with_sentiment
    ]

    return cleaned_reviews
