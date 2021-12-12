import sys
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from src.sentiment_analysis.logger import get_logger
from src.sentiment_analysis.args import parse_args
from src.sentiment_analysis.preprocess import get_processed_data
from src.sentiment_analysis.text_dataset import TextDataset
from src.sentiment_analysis.cnn import CNN
from src.sentiment_analysis.rnn import RNN
from src.sentiment_analysis.train import train_model
from src.sentiment_analysis.evaluate import evaluate_model
from src.topic_modelling.modelling import get_topics

pd.set_option("display.max_colwidth", 100)
logger = get_logger("run")


def get_train_test_dataset(train_data, test_data):
    threshold = 5
    max_len = 100
    batch_size = 32

    train_dataset = TextDataset(
        examples=train_data,
        split='train',
        threshold=threshold,
        max_len=max_len
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    test_dataset = TextDataset(
        examples=test_data,
        split='test',
        threshold=threshold,
        max_len=max_len,
        idx2word=train_dataset.idx2word,
        word2idx=train_dataset.word2idx)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )

    return train_dataset, train_loader, test_dataset, test_loader


def train_cnn(train_dataset, train_loader):
    cnn_model = CNN(vocab_size=train_dataset.vocab_size,
                    embedding_size=128,
                    output_channels=64,
                    filter_heights=[2, 3, 4],
                    stride=1,
                    dropout=0.5,
                    num_classes=2,
                    pad_idx=train_dataset.word2idx["<PAD>"]
                    )

    cnn_model = cnn_model.to(device)
    learning_rate = 5e-4
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)
    epochs = 21
    train_model(
        model=cnn_model,
        num_epochs=epochs,
        data=train_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        model_name="cnn.pt",
        device=device
    )


def train_rnn(train_dataset, train_loader):
    rnn_model = RNN(vocab_size=train_dataset.vocab_size,
                    embedding_size=128,
                    hidden_size=128,
                    num_layers=2,
                    bidirectional=True,
                    dropout=0.5,
                    num_classes=2,
                    pad_idx=train_dataset.word2idx["<PAD>"])

    rnn_model = rnn_model.to(device)
    learning_rate = 5e-4
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)
    epochs = 15
    train_model(
        model=rnn_model,
        num_epochs=epochs,
        data=train_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        model_name="rnn.pt",
        device=device
    )


def get_evaluation(model, test_data, loss_function_for_evaluation):
    return evaluate_model(model=model, data=test_data, loss_function=loss_function_for_evaluation, device=device)


def run(argv) -> None:
    args = parse_args(argv)
    logger.info(args["movie_reviews"])
    logger.info(args["operation"])

    movie_reviews_path = args["movie_reviews"]
    operation = args["operation"]

    processed_data = get_processed_data(data_path=movie_reviews_path)
    # logger.info(processed_data[0:2])
    logger.info(f"Length of data: {len(processed_data)}")

    train_data, test_data = processed_data[0:35000] + processed_data[40000: 45000], \
                            processed_data[35000: 40000] + processed_data[45000:]

    print(f"Length of Training data: {len(train_data)}, Length of Test data: {len(test_data)}")

    train_dataset, train_loader, test_dataset, test_loader = \
        get_train_test_dataset(train_data=train_data, test_data=test_data)

    if operation.lower() == "train":
        train_cnn(train_dataset=train_dataset, train_loader=train_loader)
        train_rnn(train_dataset=train_dataset, train_loader=train_loader)

    elif operation.lower() == "test":
        # rnn_model = torch.load("/Users/jaskirat/Illinois/cs-410/TISProject/sentiment-analysis/models/rnn.pt")
        # cnn_model = torch.load("/Users/jaskirat/Illinois/cs-410/TISProject/sentiment-analysis/models/cnn.pt")
        # rnn_predictions, rnn_overall_accuracy, rnn_overall_loss = \
        #     get_evaluation(
        #         model=rnn_model,
        #         test_data=test_loader,
        #         loss_function_for_evaluation=nn.CrossEntropyLoss().to(device)
        #     )
        # cnn_predictions, cnn_overall_accuracy, cnn_overall_loss = \
        #     get_evaluation(
        #         model=cnn_model,
        #         test_data=test_loader,
        #         loss_function_for_evaluation=nn.CrossEntropyLoss().to(device)
        #     )
        #
        # print(f"RNN test accuracy: {rnn_overall_accuracy / 100}, CNN test accuracy: {cnn_overall_accuracy / 100}")
        # print(f"RNN test loss: {rnn_overall_loss}, CNN test loss: {cnn_overall_loss}")
        #
        # # Evaluation on scraped IMDB reviews
        # scraped_data_path = "/Users/jaskirat/Illinois/cs-410/TISProject/sentiment-analysis/src/scraping/data/"
        # processed_scraped_data = get_processed_data(data_path=scraped_data_path)
        # print(len(processed_scraped_data))
        # train_scraped_dataset, train_scraped_loader, test_scraped_dataset, test_scraped_loader = \
        #     get_train_test_dataset(train_data=processed_scraped_data, test_data=processed_scraped_data)
        #
        # rnn_scraped_predictions, rnn_scraped_overall_accuracy, rnn_scraped_overall_loss = \
        #     get_evaluation(
        #         model=rnn_model,
        #         test_data=test_scraped_loader,
        #         loss_function_for_evaluation=nn.CrossEntropyLoss().to(device)
        #     )
        #
        # cnn_scraped_predictions, cnn_scraped_overall_accuracy, cnn_scraped_overall_loss = \
        #     get_evaluation(
        #         model=cnn_model,
        #         test_data=test_scraped_loader,
        #         loss_function_for_evaluation=nn.CrossEntropyLoss().to(device)
        #     )
        #
        # print(f"RNN test accuracy on scraped data: {rnn_scraped_overall_accuracy / 100}, "
        #       f"CNN test accuracy on scraped data: {cnn_scraped_overall_accuracy / 100}")
        # print(f"RNN test loss on scraped data: {rnn_scraped_overall_loss}, "
        #       f"CNN test loss on scraped data: {cnn_scraped_overall_loss}")

        # Topic modelling
        print("\n\nTop 20 topics for Positive reviews: ")
        get_topics(file_path="/Users/jaskirat/Illinois/cs-410/TISProject/sentiment-analysis/src/topic_modelling/data"
                             "/positive.csv")

        print("\n\nTop 20 topics for Negative reviews: ")
        get_topics(file_path="/Users/jaskirat/Illinois/cs-410/TISProject/sentiment-analysis/src/topic_modelling/data"
                             "/negative.csv")

    else:
        pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"We are using {device} as device for training!")

    run(sys.argv[1:])
