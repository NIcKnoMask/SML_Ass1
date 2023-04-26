import torch
from data_processor import CustomDataset, TestSet
from GateCNN import GateCNN
from TextCNN import TextCNN
from RNN_LSTM import LSTM
import pandas as pd
import numpy as np
from metrics import train, validate, predict, evaluation


# train and validate dataset and test set
train_data = CustomDataset(1)
validate_data = CustomDataset(domain=1, train=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=32, shuffle=False)

# initial classifier
TextCNNclf = TextCNN()
GateCNNclf = GateCNN()
LSTMclf = LSTM()

# train the model
optimizer = torch.optim.SGD(TextCNNclf.parameters(), lr=1e-2, momentum=0.9)
conv_loss, conv_acc = train(TextCNNclf, train_loader, validate_loader, optimizer)

# make prediction
test_data1 = TestSet(domain=1)
test_data2 = TestSet(domain=2)
test_loader1 = torch.utils.data.DataLoader(test_data1, batch_size=25, shuffle=False)
test_loader2 = torch.utils.data.DataLoader(test_data2, batch_size=25, shuffle=False)

# train the model for domain1
pred_domain1 = predict(TextCNNclf, test_loader1)
pred_domain1 = pred_domain1.numpy()

# use the pre-trained model to fit domain2
pred_domain2 = predict(TextCNNclf, test_loader2)
pred_domain2 = pred_domain2.numpy()

pred = np.concatenate((pred_domain1, pred_domain2), axis=None)
d = {'Id': [i for i in range(len(pred))], 'Predicted': pred}
output = pd.DataFrame(d)
output.to_csv('result.csv', index=False)
