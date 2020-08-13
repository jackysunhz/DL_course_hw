import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

# set some constants
window = 30 # time window in model training

###################
# data preparation
###################

# reading the input data
train_df = pd.read_csv('data/train.csv',parse_dates=['date'])
train_df['sales'] = train_df['sales'].values.astype(float)

train_data = train_df[train_df['date']<'2016-10-02'] # 3 years and 9 months
test_data = train_df[train_df['date']>'2016-10-01'] # last 90 days

###################
# reshape to wide
###################

def to_wide(data):

    data['id'] = 'store '+ data['store'].astype(str) + ', item ' + data['item'].astype(str)
    data = data.drop(['store','item'],axis=1)

    data_wide = data.pivot_table(index='date',
                                 columns='id',
                                 values='sales')
    return data_wide.values

train_wide = to_wide(train_data)
test_wide = to_wide(test_data)

###################
# scale to -1 to 1
###################

# scale data
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_wide.reshape(-1, train_wide.shape[-1])).reshape(train_wide.shape)
test_data_normalized = scaler.fit_transform(test_wide.reshape(-1, test_wide.shape[-1])).reshape(test_wide.shape)

###################
# get sequences
###################

def to_sequences(sequences,n_steps_in,n_steps_out):
    X, y = list(), list()

    for i in range(len(sequences)):

        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]

        X.append(seq_x)
        y.append(seq_y)

    return array(X), array(y)

train_X, train_y=to_sequences(train_data_normalized,window,1)
test_X, test_y=to_sequences(test_data_normalized,window,1)

train_X = torch.DoubleTensor(train_X) # 1340, window, 500
train_y = torch.DoubleTensor(train_y) # 1340, 1, 500
test_X = torch.DoubleTensor(test_X) # 60, window, 500
test_y = torch.DoubleTensor(test_y) # 60, 1, 500

###################
# define lstm
###################

class LSTM(nn.Module):
    def __init__(self, input_size=500, hidden_layer_size=100, output_size=500):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,window,self.hidden_layer_size),
                            torch.zeros(1,window,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out)
        return predictions[:,0,:]

def model_train(epochs = 30):
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(1,epochs+1):

        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, window, model.hidden_layer_size),
                        torch.zeros(1, window, model.hidden_layer_size))

        y_pred = model(train_X.float())

        single_loss = loss_function(y_pred.float(), train_y.view(len(train_y),500).float())
        single_loss.backward()
        optimizer.step()

        # validation
        with torch.no_grad():
            y_test_pred = model(test_X.float())
            test_loss = loss_function(y_test_pred.float(), test_y.view(len(test_y),500).float())

        if i%5 == 0:
            print(f'Epoch {i} train loss: {single_loss.item():.4f} test loss: {test_loss.item():.4f}')
    return model

model = model_train(epochs = 30)


def predict(fut_pred = 90):

    # prediction in the validation set

    test_inputs = train_X[-1][-window:]

    model.eval()

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-window:].float()).view(1,-1,500)
        with torch.no_grad():
            model.hidden = (torch.zeros(1, window, model.hidden_layer_size),
                            torch.zeros(1, window, model.hidden_layer_size))
            test_inputs = torch.cat((test_inputs.float(),model(seq)),0)

    actual_predictions = scaler.inverse_transform(np.array(test_inputs[window:] ))
    return actual_predictions

actual_predictions = predict(fut_pred = 90)
pred_1_1 = actual_predictions[:,0]

observed = test_data['sales'][test_data['id']=='store 1, item 1'].values

# plot calibration
plt.ylabel('Sales')
plt.grid(True)
plt.autoscale(axis='x', tight=True)

plt.plot(observed,label="observed")
plt.plot(pred_1_1,label="predicted")
plt.legend()
plt.show()

# rmse
math.sqrt(mean_squared_error(pred_1_1,observed)) # 5.94









