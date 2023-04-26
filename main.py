import torch
from data_processor import CustomDataset, TestSet
from GateCNN import GateCNN
from TextCNN import TextCNN
from RNN_LSTM import LSTM
import pandas as pd
from metrics import train, validate, predict, evaluation


# train and validate dataset and test set
train_data = CustomDataset(1)
validate_data = CustomDataset(domain=1, train=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=32, shuffle=False)

# The data that used to predict toe final result in kaggle
test_data = TestSet()
test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=False)

# initial classifier
TextCNNclf = TextCNN()
GateCNNclf = GateCNN()
LSTMclf = LSTM()

# train the model
optimizer = torch.optim.SGD(TextCNNclf.parameters(), lr=1e-2, momentum=0.9)
conv_loss, conv_acc = train(TextCNNclf, train_loader, validate_loader, optimizer)

# make prediction
pred = predict(TextCNNclf, test_loader)

# output the prediction as .csv
pred = pd.Series(pred)
d = {'Id': [i for i in range(len(pred))], 'Predicted': pred}
output = pd.DataFrame(d)
output.to_csv('result.csv', index=False)
