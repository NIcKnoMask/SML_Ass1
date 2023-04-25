import torch
import time
import pandas as pd
from data_processor import CustomDataset
from data_processor import TestSet
from CNN import Conv1dModel


def validate(model, criterion, test_loader):
    test_loss = 0.
    test_preds, test_labels = list(), list()
    for i, data in enumerate(test_loader):
        x, labels = data

        with torch.no_grad():
            logits = model(x)  # Compute scores
            predictions = torch.argmax(logits, dim=1)
            test_loss += criterion(input=logits, target=labels).item()
            test_preds.append(predictions)
            test_labels.append(labels)

    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)

    test_accuracy = torch.eq(test_preds, test_labels).float().mean().item()

    print('[TEST] Mean loss {:.4f} | Accuracy {:.4f}'.format(test_loss/len(test_loader), test_accuracy))


def train(model, train_loader, validate_loader, optimizer, n_epochs=10):
    LOG_INTERVAL = 64
    running_loss, running_accuracy = list(), list()
    start_time = time.time()
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):  # Loop over training dataset `n_epochs` times

        epoch_loss = 0.

        for i, data in enumerate(train_loader):  # Loop over elements in training set

            x, labels = data

            logits = model(x)

            predictions = torch.argmax(logits, dim=1)
            train_acc = torch.mean(torch.eq(predictions, labels).float()).item()

            loss = criterion(input=logits, target=labels)

            loss.backward()  # Backward pass (compute parameter gradients)
            optimizer.step()  # Update weight parameter using SGD
            optimizer.zero_grad()  # Reset gradients to zero for next iteration

            running_loss.append(loss.item())
            running_accuracy.append(train_acc)

            epoch_loss += loss.item()

            if i % LOG_INTERVAL == 0:  # Log training stats
                deltaT = time.time() - start_time
                mean_loss = epoch_loss / (i + 1)
                print(len(data), len(data[0]))
                print('[TRAIN] Epoch {} [{}/{}]| Mean loss {:.4f} | Train accuracy {:.5f} | Time {:.2f} s'.format(epoch, i, len(train_loader), mean_loss, train_acc,deltaT))

        print('Epoch complete! Mean loss: {:.4f}'.format(epoch_loss / len(train_loader)))
        validate(model, criterion, validate_loader)

    return running_loss, running_accuracy


def predict(model, test_loader):
    test_preds = list()
    for i, data in enumerate(test_loader):
        x, labels = data

        with torch.no_grad():
            logits = model(x)  # Compute scores
            predictions = torch.argmax(logits, dim=1)
            test_preds.append(predictions)
    test_preds = torch.cat(test_preds)
    return test_preds


human_path = "data/set1_human.json"
machine_path = "data/set1_machine.json"
human_path2 = "data/set2_human.json"
machine_path2 = "data/set2_machine.json"
test_path = "data/test.json"

# train and validate dataset and test set
train_data = CustomDataset(1, [3500, 3500, 100, 100], train=True)
validate_data = CustomDataset(1, [3500, 3500, 100, 100], train=False)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=32, shuffle=False)

test_data = TestSet(test_path)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# initial classifier
clf = Conv1dModel(5000)

# train the model
optimizer = torch.optim.SGD(clf.parameters(), lr=1e-2, momentum=0.9)
conv_loss, conv_acc = train(clf, train_loader, validate_loader, optimizer)

# make prediction
pred = predict(clf, test_loader)

# output the prediction as .csv
pred = pd.Series(pred)
d = {'Id': [i for i in range(len(pred))], 'Predicted': pred}
output = pd.DataFrame(d)
output.to_csv('result_CNN.csv', index=False)