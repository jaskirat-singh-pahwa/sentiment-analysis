from src.sentiment_analysis.args import parse_args


def test_parse_args() -> None:
    test_args = [
        "--movie-reviews", "users/jaskirat/movie_reviews.csv",
        "--operation", "train"
    ]
    args = parse_args(test_args)

    assert args == {
        "movie_reviews": "users/jaskirat/movie_reviews.csv",
        "operation": "train"
    }
