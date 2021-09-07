import torch.nn as nn
from torchsummary import summary

'''
activation_function : ReLU

conv2d 5x5 in 3 out 8 stride 1
maxpooling2d stride 2

conv2d 5x5 in 8  out 16 stride 1
maxpooling2d stride 2

conv2d 5x5 in 16 out 120 

dropout 0.25
fcn
dense
dropout 0.5 
dense

optimizer : Adam
loss : categorical cross entropy
'''

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(8, affine=True, momentum=0.99, eps=1e-3)
        self.relu = nn.ReLU()
        self.Maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16, momentum=0.99, eps=1e-3)
        self.relu2 = nn.ReLU()
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.25)
        self.flatten = nn.Flatten()
        self.FC = nn.Linear(17280 , 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.FC2 =nn.Linear(128, 7)


    def forward(self, input):
        x = input
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.Maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.Maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.FC(x)
        x = self.dropout2(x)
        x = self.FC2(x)

        return x

Model = Model()
print(Model)
print(summary(Model, (3, 48, 48), device = 'cpu'))
