import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # ==========================
        # TODO 1: build your network
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(in_features=(64 * 8 * 8), out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        # ==========================

    def forward(self, x):
        # (batch_size, 3, 256, 256)

        # ========================
        # TODO 2: forward the data
        # please write down the output size of each layer
        # example:
        # out = self.relu(self.conv1(x))
        # (batch_size, 64, 256, 256)

        out = self.relu(self.conv1(x))
        # (batch_size, 32, 256, 256)

        out = self.relu(self.conv3(out))

        out = self.pool(out)

        out = self.relu(self.conv2(out))
        # (batch_size, 64, 256, 256)

        out = self.relu(self.conv4(out))

        out = self.pool(out)
        # (batch_size, 64, 128, 128)

        out = torch.flatten(out, 1)
        # TODO 4: fully connected layers
        out = self.fc1(out)
        out = self.fc2(out)
        
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

    train_acc = train_acc/len(train_loader)
    train_loss = train_loss/len(train_loader)

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
            valid_acc += calc_acc(output, target)
            valid_loss += criterion(output, target)

            # probabilities = torch.exp(output)
            # top_prob, top_class = probabilities.topk(1, dim=1)
            # predictions = top_class == target.view(*top_class.shape)
            # valid_acc += torch.mean(predictions.type(torch.FloatTensor))


            # ================================

    valid_acc = valid_acc/len(valid_loader)
    valid_loss = valid_loss.item()/len(valid_loader)

    return valid_acc, valid_loss


def main():
    # torch.manual_seed(0)
    # ==================
    # TODO 9: set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ==================


    # ========================
    # TODO 10: hyperparameters
    # you can add your parameters here
    LEARNING_RATE = 0.01
    BATCH_SIZE = 50
    EPOCHS = 100

    # TRAIN_DATA_PATH = './resized2/train'
    # VALID_DATA_PATH = './resized2/valid'

    TRAIN_DATA_PATH = '/content/drive/MyDrive/resized2/train'
    VALID_DATA_PATH = '/content/drive/MyDrive/resized2/valid'
    MODEL_PATH = 'hanwritten-digit.pt'
    MODEL_PATH = '/content/drive/MyDrive/data/hanwritten-digit_雙層.pt'

    # ========================


    # ===================
    # TODO 11: transforms
    train_transform = transforms.Compose([
        # may be adding some data augmentations?
        transforms.Resize(36),
        transforms.RandomResizedCrop(32, (0.3, 0.6)),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # ===================
    valid_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    # =================
    # TODO 12: set up datasets
    # hint: ImageFolder?
    train_data = datasets.ImageFolder(TRAIN_DATA_PATH, transform=valid_transform)
    train_data2 = datasets.ImageFolder(TRAIN_DATA_PATH, transform=train_transform)
    valid_data = datasets.ImageFolder(VALID_DATA_PATH, transform=valid_transform)

    #切分80%當作訓練集、20%當作驗證集
    # train_size = int(0.8 * len(valid_data))
    # valid_size = len(valid_data) - train_size

    # train_data2 = torch.utils.data.Subset(valid_data, range(train_size))
    # valid_data = torch.utils.data.Subset(valid_data, range(train_size, len(valid_data)))
    # train_data = torch.utils.data.Subset(train_data, range(train_size))

    # train_data, _ = torch.utils.data.random_split(train_data, [train_size, valid_size])
    # train_data2, valid_data = torch.utils.data.random_split(valid_data, [train_size, valid_size])
    
    train_data = torch.utils.data.ConcatDataset([train_data, train_data2])
    
    # train_data = datasets.ImageFolder(TRAIN_DATA_PATH, transform=train_transform)
    # valid_data = datasets.ImageFolder(VALID_DATA_PATH, transform=valid_transform)

    # =================


    # ============================
    # TODO 13 : set up dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    train_loader2 = torch.utils.data.DataLoader(train_data2, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
    # ============================

    # check transfrom image
    # dataiter = iter(train_loader)
    # dataiter2 = iter(train_loader2)

    # images, labels = dataiter.next()
    # images2, labels2 = dataiter2.next()

    # images = images.numpy()
    # images2 = images2.numpy()

    # #plot
    # fig = plt.figure(figsize=(25,4))

    # for idx in np.arange(20):
    #     ax = fig.add_subplot(1, 20, idx+1, xticks=[], yticks=[])
    #     plt.imshow(np.transpose(images[idx] * 255, (1,2,0)))

    # fig2 = plt.figure(figsize=(25,4))

    # for idx in np.arange(20):
    #     ax = fig2.add_subplot(1, 20, idx+1, xticks=[], yticks=[])
    #     plt.imshow(np.transpose(images2[idx] * 255, (1,2,0)))
    # ================================

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

    # print('Start training...')
    # for epoch in range(EPOCHS):
    #     print(f'epoch {epoch} start...')

    #     train_acc[epoch], train_loss[epoch] = training(model, device, train_loader, criterion, optimizer)
    #     valid_acc[epoch], valid_loss[epoch] = validation(model, device, valid_loader, criterion)

    #     print(f'epoch={epoch} train_acc={train_acc[epoch]} train_loss={train_loss[epoch]} valid_acc={valid_acc[epoch]} valid_loss={valid_loss[epoch]}')
    # print('Training finished')


    # # ==================================
    # # TODO 15: save the model parameters
    # torch.save(model, MODEL_PATH)
    # # ==================================


    # # ========================================
    # # TODO 16: draw accuracy and loss pictures
    # # lab2_teamXX_accuracy.png, lab2_teamXX_loss.png
    # # hint: plt.plot
    # plt.figure("loss")
    # plt.plot(train_loss, label='Training loss')
    # plt.plot(valid_loss, label='Valid loss')
    # plt.legend()

    # plt.figure("acc")
    # plt.plot(train_acc, label='Training acc')
    # plt.plot(valid_acc, label='Valid acc')
    # plt.legend()
    # plt.show()


    # =========================================
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/files/', train=False, download=True,
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])),
        batch_size=BATCH_SIZE, shuffle=True)
    model = torch.load(MODEL_PATH)
    valid_acc, valid_loss = validation(model, device, test_loader, criterion)
    print(f'valid_acc={valid_acc} valid_loss={valid_loss}')


if __name__ == '__main__':
    main()