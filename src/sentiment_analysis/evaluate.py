import torch
from src.sentiment_analysis.logger import get_logger

logger = get_logger(name="evaluate")


def get_accuracy(output, labels):
    predictions = output.argmax(dim=1)
    correct_predictions = (predictions == labels).sum().float()
    accuracy = 100 * correct_predictions / len(labels)

    return accuracy


def evaluate_model(model, data, loss_function, device):
    logger.info("Evaluating model performance on the test data: ")
    model.eval()

    total_epoch_loss = 0
    total_epoch_accuracy = 0
    predictions = []
    for text_reviews, labels in data:
        text_reviews = text_reviews.to(device)
        labels = labels.to(device)

        output = model(text_reviews)
        accuracy = get_accuracy(output, labels)
        prediction = output.argmax(dim=1)
        predictions.append(prediction)

        loss = loss_function(output, labels)
        total_epoch_loss += loss.item()
        total_epoch_accuracy += accuracy.item()

    overall_accuracy = 100 * total_epoch_accuracy / len(data)
    overall_loss = total_epoch_loss / len(data)

    print('[TEST]\t Loss: {:.4f}\t Accuracy: {:.2f}%'.format(overall_loss, overall_accuracy))
    predictions = torch.cat(predictions)

    return predictions, overall_accuracy, overall_loss
