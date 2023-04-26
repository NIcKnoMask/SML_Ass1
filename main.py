import torch
from data_processor import CustomDataset, TestSet
from GateCNN import GateCNN
from TextCNN import TextCNN
from RNN_LSTM import LSTM
import pandas as pd
import numpy as np
from metrics import train, validate, predict, evaluation


# train and validate dataloader initial
domain1_train_data = CustomDataset(1)
domain1_validate_data = CustomDataset(domain=1, train=False)
domain2_train_data = CustomDataset(2)
domain2_validate_data = CustomDataset(domain=2, train=False)

domain1_train_loader = torch.utils.data.DataLoader(domain1_train_data, batch_size=64, shuffle=True)
domain1_validate_loader = torch.utils.data.DataLoader(domain1_validate_data, batch_size=64, shuffle=False)
domain2_train_loader = torch.utils.data.DataLoader(domain2_train_data, batch_size=25, shuffle=True)
domain2_validate_loader = torch.utils.data.DataLoader(domain2_validate_data, batch_size=25, shuffle=False)

# initial classifier
TextCNNclf = TextCNN()
GateCNNclf = GateCNN()
LSTMclf = LSTM()

# initial test data
test_data1 = TestSet(domain=1)
test_data2 = TestSet(domain=2)
test_loader1 = torch.utils.data.DataLoader(test_data1, batch_size=25, shuffle=False)
test_loader2 = torch.utils.data.DataLoader(test_data2, batch_size=25, shuffle=False)

# train the model for domain1
optimizer = torch.optim.SGD(TextCNNclf.parameters(), lr=1e-2, momentum=0.9)
conv_loss, conv_acc = train(TextCNNclf, domain1_train_loader, domain1_validate_loader, optimizer)

# use the model to predict domain1 test data
pred_domain1 = predict(TextCNNclf, test_loader1)
pred_domain1 = pred_domain1.numpy()

# fine-tuning the domain2 data
optimizer = torch.optim.SGD(TextCNNclf.parameters(), lr=1e-2, momentum=0.9)
conv_loss, conv_acc = train(TextCNNclf, domain2_train_loader, domain2_validate_loader, optimizer)

# use the tuned model to predict domain2 data
pred_domain2 = predict(TextCNNclf, test_loader2)
pred_domain2 = pred_domain2.numpy()

# concatenate domain1 and domain2 prediction then output
pred = np.concatenate((pred_domain1, pred_domain2), axis=None)
d = {'Id': [i for i in range(len(pred))], 'Predicted': pred}
output = pd.DataFrame(d)
output.to_csv('result.csv', index=False)
