import torch.nn as nn
from collections import OrderedDict


class CNN(nn.Module):
    def __init__(self, in_planes, action_space, planes=32, value="state"):
        super(CNN, self).__init__()
        if value == "state":
            action_space = 1

        self.cnn = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(planes, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.output = nn.Linear(4096, action_space)

        for name, para in self.named_parameters():
            print({name: [para.shape, para.requires_grad]})

    def forward(self, x):
        x = x / 255
        x = self.cnn(x)
        return self.output(x)





class MLP(nn.Module):
    def __init__(self, input_dim, action_space, hidden_dim=128, value="state", layer=2):
        super(MLP, self).__init__()
        if value == "state":
            action_space = 1

        layer_dims = [input_dim] + [hidden_dim] * layer
        self.layers = nn.Sequential(
            OrderedDict(
                {
                    f"Hidden_layer{i}": nn.Sequential(
                        nn.Linear(layer_dims[i], layer_dims[i + 1]), #nn.BatchNorm1d(layer_dims[i + 1]),
                        nn.ReLU()
                    ) for i in range(len(layer_dims) - 1)
                }
            )
        )
        self.output_layer = nn.Linear(layer_dims[-1], action_space)

        for name, para in self.named_parameters():
            print({name: [para.shape, para.requires_grad]})

    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x)


