import pandas as pd


def clean_scraped_reviews(reviews_path):
    with open(reviews_path, "r") as file:
        reviews = file.readlines()

    print(len(reviews))
    clean_reviews = []
    for review in reviews:
        clean_reviews.append(review.replace('"', '').replace("'", "").split(",", 3)[3].strip())

    df = pd.DataFrame(data=clean_reviews, columns=["reviews"])
    df.to_csv("./clean_scraped_reviews.csv", index=False)


if __name__ == "__main__":
    scraped_reviews_path = "./scraped_reviews.csv"

    clean_scraped_reviews(reviews_path=scraped_reviews_path)
