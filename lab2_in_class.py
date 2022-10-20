import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # TODO 1: convolution layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # TODO 2: fully connected layers
        self.fc1 = nn.Linear(in_features=(512 * 7 * 7), out_features=4096)
        self.fc2 = nn.Linear(in_features=(4096), out_features=4096)
        self.fc3 = nn.Linear(in_features=(4096), out_features=1000)
        # 最終辨認的種類為1000類

    def forward(self, x):
        # TODO 3: convolution and pooling layers
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))

        out = self.pool(out)

        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))

        out = self.pool(out)

        out = self.relu(self.conv5(out))
        out = self.relu(self.conv6(out))
        out = self.relu(self.conv7(out))

        out = self.pool(out)

        out = self.relu(self.conv8(out))
        out = self.relu(self.conv9(out))
        out = self.relu(self.conv10(out))

        out = self.pool(out)

        out = self.relu(self.conv11(out))
        out = self.relu(self.conv12(out))
        out = self.relu(self.conv13(out))

        out = self.pool(out)

        out = torch.flatten(out, 1)
        # TODO 4: fully connected layers
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 10

    model = Net().to(device)
    data = torch.rand(batch_size, 3, 224, 224).to(device)
    output = model(data)
    print(output.shape)

if __name__ == '__main__':
    main()