from src.sentiment_analysis.preprocess import (
    remove_special_chars_from_words,
    get_processed_data
)


def test_remove_special_chars_from_words() -> None:
    review = "(I liked the movie) a; lo!t"
    actual = remove_special_chars_from_words(review=review)
    expected = ["(", "I", "liked", "the", "movie", ")", "a", ";", "lo!t"]

    assert actual == expected


def test_get_processed_data() -> None:
    input_reviews_path = "test/sentiment_analysis/test_data/"
    # input_reviews = pd.read_csv(input_reviews_path, index_col=None, header=0)
    actual = get_processed_data(data_path=input_reviews_path)
    """
    This movie is one of the amazing movies I have ever seen,1
    This movie is worst) Its not worth the 'time,0
    (I love the) acting of all the actors,1
    (Thi(s is! b&y fa.r the best movie of Dwayne Johnson,1
    I don't like the movie....) at al!!!!!!!l,0
    """
    expected = [
        ("1",  ["This", "movie", "is", "one", "of", "the", "amazing", "movies", "I", "have", "ever", "seen"]),
        ("0", ["This", "movie", "is", "worst", ")", "Its", "not", "worth", "the", "'", "time"]),
        ("1", ["(", "I", "love", "the", ")", "acting", "of", "all", "the", "actors"]),
        ("1", ["(", "Thi(s", "is", "!", "b&y", "fa.r", "the", "best", "movie", "of", "Dwayne", "Johnson"]),
        ("0", ["I", "don't", "like", "the", "movie....", ")", "at", "al!!!!!!!l"])
    ]

    assert actual == expected
