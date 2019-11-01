import torch
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Load data(do not change)
data = pd.read_csv("../data/mnist_train.csv")
train_data = data[:2000]
test_data = data[2000:2500]

# ----- Prepare Data ----- #
# step one: preparing your data including data normalization
train_X = train_data.iloc[:, 1:]
train_Y = train_data["label"]
test_X = test_data.iloc[:, 1:]
test_Y = test_data["label"]

min_max_scaler = preprocessing.MinMaxScaler()
train_X_data_norm = min_max_scaler.fit_transform(train_X)
test_X_data_norm = min_max_scaler.fit_transform(test_X)

# step two: transform np array to pytorch tensor
train_X_tensor = torch.tensor(train_X_data_norm, dtype=torch.float32).reshape(-1, 1, 28, 28)
test_X_tensor = torch.tensor(test_X_data_norm, dtype=torch.float32).reshape(-1, 1, 28, 28)
train_Y_tensor = torch.tensor(train_Y)
test_Y_tensor = torch.tensor(test_Y.to_numpy())


# ----- Build CNN Network ----- #
# Define your model here
class mymodel(torch.nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)


# Define our model
model = mymodel()
# Define your learning rate
learning_rate = 0.01
# Define your optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Define your loss function
criterion = torch.nn.NLLLoss()


# ----- Complete PlotLearningCurve function ----- #
def PlotLearningCurve(epoch, trainingloss, testingloss):
    plt.plot(epoch, trainingloss, color='blue')
    plt.plot(epoch, testingloss, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('../OutputImages/Deep_Learning/lossvsepochs.png')
    plt.show()


# ----- Main Function ----- #
trainingloss = []
testingloss = []
# Define number of iterations
epochs = 100
for epoch in range(1, epochs + 1):
    model.train()
    # step one : fit your model by using training data and get predict label
    output = model(train_X_tensor)
    # step two: calculate your training loss
    loss = criterion(output, train_Y_tensor)
    # step three: calculate backpropagation
    optimizer.zero_grad()
    # step four: update parameters
    loss.backward()
    # step five: reset our optimizer
    optimizer.step()
    # step six: store your training loss
    trainingloss += loss.item(),
    # step seven: evaluation your model by using testing data and get the accuracy
    correct = 0
    with torch.no_grad():
        total = 0
        model.eval()
        # predict testing data
        output_test = model(test_X_tensor)
        # calculate your testing loss
        loss = criterion(output_test, test_Y_tensor)
        # store your testing loss
        testingloss += loss.item(),
        if epoch % 10 == 0:
            # get labels with max values
            _, predicted = torch.max(output_test, 1)
            total += test_Y_tensor.size(0)
            correct += (predicted == test_Y_tensor).sum().item()
            acc = 100 * correct / total
            print('Epoch:', epoch, 'Test Accuracy:', acc)

PlotLearningCurve(range(len(trainingloss)), trainingloss, testingloss)
