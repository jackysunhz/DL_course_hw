import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# import data
train_data = pd.read_csv('Data/train.csv') # 1022 by 258
test_data = pd.read_csv('Data/test.csv') # 438 by 257

# find a subset of variables that are balanced
to_use = list(range(2,122)) + list(range(129,150)) +\
         list(range(170,200)) + list(range(210,258))

# training set
train_x = train_data.iloc[:,to_use ]
train_mean = train_x.mean()
train_std = train_x.std()
train_x = (train_x - train_mean)/train_std

train_X = torch.tensor(train_x.values).float()

train_Y = torch.tensor(train_data.iloc[:,1].values).float().view(-1,1)
train_Y = torch.log(train_Y)

test_x = test_data.iloc[:,[x - 1 for x in to_use]]
test_x = (test_x - train_mean)/train_std

test_X = torch.tensor(test_x.values).float()

# prepare training data loader
train_ds = TensorDataset(train_X,train_Y)

batch_size = 32
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# define model
class bmodel(nn.Module):
    def __init__(self, input_d, hidden_d, output_d):
        super().__init__()
        self.fc1 = nn.Linear(input_d,hidden_d)
        self.fc2 = nn.Linear(hidden_d,hidden_d)
        self.fc3 = nn.Linear(hidden_d,output_d)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# define dimensions
input_d = len(to_use)
hidden_d = 1024
output_d = 1

model = bmodel(input_d, hidden_d, output_d)

#Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

#Define loss function
loss_fn = F.mse_loss

#Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(1,num_epochs+1):
        for xb,yb in train_dl:
            opt.zero_grad()
            #Generate predictions
            pred = model(xb)
            loss = loss_fn(pred,yb)
            #Perform gradient descent
            loss.backward()
            opt.step()
        if epoch % 50 == 0:
            print("[%d/%d] loss: %0.4f" % (epoch,num_epochs,loss))
    #print('Training loss: ', loss_fn(model(train_X), train_y))

#Train the model for 10 epochs
fit(500, model, loss_fn, opt)

preds = torch.exp(model(train_X))
import matplotlib.pyplot as plt

plt.plot(torch.exp(train_Y).detach().numpy(), preds.detach().numpy(), 'o')
plt.xlabel('Targeted y', fontsize=16)
plt.ylabel('Modeled y', fontsize=16)
plt.ylim(1e5,8e5)
plt.xlim(1e5,8e5)
plt.show()

# calculate training MSE for NN
((train_Y - preds).pow(2)).mean() # 1.61e11

# predict in the test set
with torch.no_grad():
    output = torch.exp(model(test_X))

out = {'Id': test_data.iloc[:,0],
       'SalePrice': output.squeeze().tolist()}

out_d = pd.DataFrame(out)

out_d.to_csv("Data/sample_nn_test.csv",index=False)

# HOW ABOUT LASSO
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error

train_x = train_data.iloc[:,2:]
train_y = train_data.iloc[:,1]

lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(train_x, train_y)

lasso = Lasso(max_iter = 10000, normalize = True)
lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(train_x, train_y)
mean_squared_error(train_y, lasso.predict(train_x)) # 6.76e8

# plot predictions
pp = lasso.predict(train_x)
plt.plot(train_y, pp, 'o')
plt.xlabel('Targeted y', fontsize=12)
plt.ylabel('Modeled y', fontsize=12)
plt.ylim(1e5,8e5)
plt.xlim(1e5,8e5)
plt.show()

#Generate predictions for the test set
test_x = test_data.iloc[:,1:]
y_out = lasso.predict(test_x)

out = {'Id': test_data.iloc[:,0],
       'SalePrice': y_out}

out_d = pd.DataFrame(out)

out_d.to_csv("Data/sample_lasso_test.csv",index=False)


