import torch.nn as nn
import torch.nn.functional as F
class simpleConvNet(nn.Module):
    def __init__(self):
        super(simpleConvNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.filterNr = 16
        self.conv1 = nn.Conv2d(1, self.filterNr, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.filterNr, self.filterNr*2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.filterNr*2, self.filterNr*2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.filterNr*2, self.filterNr*4, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(self.filterNr*4, self.filterNr*4, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(256 * self.filterNr*4, 28,bias=True)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 256 * self.filterNr*4)
        x = self.fc1(x)
        return x

