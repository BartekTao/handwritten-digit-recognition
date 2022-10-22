import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # ==========================
        # TODO 1: build your network



        # ==========================


    def forward(self, x):
        # (batch_size, 3, 256, 256)

        # ========================
        # TODO 2: forward the data
        # please write down the output size of each layer
        # example:
        # out = self.relu(self.conv1(x))
        # (batch_size, 64, 256, 256)
        


        # ========================
        return out


def calc_acc(output, target):
    predicted = torch.max(output, 1)[1]
    num_samples = target.size(0)
    num_correct = (predicted == target).sum().item()
    return num_correct / num_samples


def training(model, device, train_loader, criterion, optimizer):
    # ===============================
    # TODO 3: switch the model to training mode
    model.train()
    # ===============================
    train_acc = 0.0
    train_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        # =============================================
        # TODO 4: initialize optimizer to zero gradient
        optimizer.zero_grad()
        # =============================================

        output = model(data)

        # =================================================
        # TODO 5: loss -> backpropagation -> update weights
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # =================================================

        train_acc += calc_acc(output, target)
        train_loss += loss.item()

    train_acc /= len(train_loader)
    train_loss /= len(train_loader)

    return train_acc, train_loss


def validation(model, device, valid_loader, criterion):
    # ===============================
    # TODO 6: switch the model to validation mode
    model.eval()
    # ===============================
    valid_acc = 0.0
    valid_loss = 0.0

    # =========================================
    # TODO 7: turn off the gradient calculation
    with torch.no_grad():
    # =========================================
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            # ================================
            # TODO 8: calculate accuracy, loss
            log_probabilities = model(data)
            valid_loss += criterion(log_probabilities, target)

            probabilities = torch.exp(log_probabilities)
            top_prob, top_class = probabilities.topk(1, dim=1)
            predictions = top_class == target.view(*top_class.shape)
            valid_acc += torch.mean(predictions.type(torch.FloatTensor))


            # ================================

    valid_acc /= len(valid_loader)
    valid_loss /= len(valid_loader)

    return valid_acc, valid_loss


def main():
    # ==================
    # TODO 9: set device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    # ==================


    # ========================
    # TODO 10: hyperparameters
    # you can add your parameters here
    LEARNING_RATE = 0.01
    BATCH_SIZE = 50
    EPOCHS = 10
    TRAIN_DATA_PATH = './data/train/'
    VALID_DATA_PATH = './data/valid/'
    MODEL_PATH = 'hanwritten-digit.pt'

    # ========================


    # ===================
    # TODO 11: transforms
    train_transform = transforms.Compose([
        # may be adding some data augmentations?

        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # ===================
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    # =================
    # TODO 12: set up datasets
    # hint: ImageFolder?
    train_data = datasets.MNIST(TRAIN_DATA_PATH, download=True, train=True, transform=train_transform)
    valid_data = datasets.MNIST(VALID_DATA_PATH, download=True, train=False, transform=valid_transform)
    # =================


    # ============================
    # TODO 13 : set up dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
    # ============================

    # build model, criterion and optimizer
    model = Net().to(device).train()
    # ================================
    # TODO 14: criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # ================================


    # training and validation
    train_acc = [0.0] * EPOCHS
    train_loss = [0.0] * EPOCHS
    valid_acc = [0.0] * EPOCHS
    valid_loss = [0.0] * EPOCHS

    print('Start training...')
    for epoch in range(EPOCHS):
        print(f'epoch {epoch} start...')

        train_acc[epoch], train_loss[epoch] = training(model, device, train_loader, criterion, optimizer)
        valid_acc[epoch], valid_loss[epoch] = validation(model, device, valid_loader, criterion)

        print(f'epoch={epoch} train_acc={train_acc[epoch]} train_loss={train_loss[epoch]} valid_acc={valid_acc[epoch]} valid_loss={valid_loss[epoch]}')
    print('Training finished')


    # ==================================
    # TODO 15: save the model parameters
    torch.save(model, MODEL_PATH)
    # ==================================


    # ========================================
    # TODO 16: draw accuracy and loss pictures
    # lab2_teamXX_accuracy.png, lab2_teamXX_loss.png
    # hint: plt.plot
    plt.plot(train_loss, label='Training loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()
    plt.grid()


    # =========================================


if __name__ == '__main__':
    main()