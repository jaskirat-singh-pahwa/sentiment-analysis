import torch
from src.sentiment_analysis.constants import Constants
from src.sentiment_analysis.logger import get_logger

logger = get_logger(name="train")
device = Constants.device


def get_accuracy(output, labels):
    predictions = output.argmax(dim=1)
    correct_predictions = (predictions == labels).sum().float()
    accuracy = 100 * correct_predictions / len(labels)

    return accuracy


def train_model(model, num_epochs, data, loss_function, optimizer, model_name, device):
    logger.info("Training model:")
    model.train()

    for epoch in range(num_epochs):
        total_epoch_loss = 0
        total_epoch_accuracy = 0
        for text_reviews, labels in data:
            text_reviews = text_reviews.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(text_reviews)
            accuracy = get_accuracy(output, labels)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            total_epoch_accuracy += accuracy.item()

        logger.info(
            "Train ---\t Epoch: {:2d}\t Loss: {:.3f}\t Train Accuracy: {:.3f}%"
            .format(
                epoch + 1,
                total_epoch_loss / len(data),
                100 * total_epoch_accuracy / len(data)
            )
        )
    torch.save(model, "/Users/jaskirat/Illinois/cs-410/TISProject/sentiment-analysis/models/" + model_name)
    logger.info("Model trained!")
