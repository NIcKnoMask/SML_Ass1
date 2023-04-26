import torch
import time
import pandas as pd
from torchmetrics.classification import BinaryRecall, BinaryPrecision, BinaryConfusionMatrix
from data_processor import CustomDataset
from data_processor import TestSet
from GateCNN import GateCNN


def validate(model, criterion, test_loader):
    test_loss = 0.
    test_preds, test_labels = list(), list()
    for i, data in enumerate(test_loader):
        x, labels = data

        with torch.no_grad():
            y_output = model(x)  # Compute scores
            predictions = torch.argmax(y_output, dim=1)
            test_loss += criterion(input=y_output, target=labels).item()
            test_preds.append(predictions)
            test_labels.append(labels)

    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)

    valid_accuracy = torch.eq(test_preds, test_labels).float().mean().item()
    recall_metric = BinaryRecall()
    valid_recall = recall_metric(test_preds, test_labels)
    precision_metric = BinaryPrecision()
    valid_precision = precision_metric(test_preds, test_labels)
    bcm = BinaryConfusionMatrix()
    valid_cfmatrix = bcm(test_preds, labels)

    return valid_accuracy, valid_precision, valid_recall, valid_cfmatrix, test_loss


def train(model, train_loader, validate_loader, optimizer, n_epochs=10):
    LOG_INTERVAL = 64
    running_loss, running_accuracy = list(), list()
    start_time = time.time()
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):  # Loop over training dataset `n_epochs` times
        epoch_loss = 0.
        for i, data in enumerate(train_loader):  # Loop over elements in training set
            x, labels = data
            y_output = model(x)
            predictions = torch.argmax(y_output, dim=1)

            # evaluate the prediction
            train_acc = torch.mean(torch.eq(predictions, labels).float()).item()  # accuracy rate
            recall_metric = BinaryRecall()
            train_recall = recall_metric(predictions, labels)
            precision_metric = BinaryPrecision()
            train_precision = precision_metric(predictions, labels)

            # backward process
            loss = criterion(input=y_output, target=labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss.append(loss.item())
            running_accuracy.append(train_acc)

            epoch_loss += loss.item()

            if i % LOG_INTERVAL == 0:  # Log training stats
                deltaT = time.time() - start_time
                mean_loss = epoch_loss / (i + 1)
                print(len(data), len(data[0]))
                print('[TRAIN] Epoch {} [{}/{}]| Mean loss {:.4f} | Train accuracy {:.5f} | Train precision {:.5f} '
                      'Train recall {:.5f} |Time {:.2f} s'.format(epoch, i, len(train_loader), mean_loss, train_acc,
                                                                  train_precision, train_recall, deltaT))
        print('Epoch complete! Mean loss: {:.4f}'.format(epoch_loss / len(train_loader)))

        valid_accuracy, valid_precision, valid_recall, _, valid_loss = validate(model, criterion, validate_loader)
        print('[TEST] Mean loss {:.4f} | Accuracy {:.4f}'.format(valid_loss / len(validate_loader), valid_accuracy))
    return running_loss, running_accuracy


def predict(model, test_loader):
    test_preds = list()
    for i, data in enumerate(test_loader):
        x, labels = data
        with torch.no_grad():
            y_output = model(x)  # Compute scores
            predictions = torch.argmax(y_output, dim=1)
            test_preds.append(predictions)
    test_preds = torch.cat(test_preds)
    return test_preds


def evaluation(test_data, model):
    criterion = torch.nn.CrossEntropyLoss()
    valid_accuracy, valid_precision, valid_recall, valid_conf_matrix, valid_loss = validate(model, criterion, test_data)
    print(f'test accuracy: {valid_accuracy} | test precision: {valid_precision} | test recall: {valid_recall}')
    print(valid_conf_matrix)
