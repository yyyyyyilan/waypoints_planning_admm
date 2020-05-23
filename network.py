import torch
from torch import nn

class LinearNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        """Network structure is defined here
        """
        super(LinearNetwork, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions

        self.fc_in = nn.Linear(self.input_size, 64)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64, 256)
        self.fc_out = nn.Linear(256, self.num_actions)
    
    def forward(self, s_input):
        x = self.fc_in(s_input)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x


class ConvNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        """Network structure is defined here
        """
        super(ConvNetwork, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions

        self.state_cnn = nn.Sequential(
            # nn.BatchNorm3d(1),
            nn.Conv3d(1, 128, 3, stride=1),
            # nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, 3, stride=1),
            # nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, 2, stride=1),
            # nn.BatchNorm3d(256),
            nn.ReLU(),
        )
        self.state_model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
        )
        self.loc_model = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(512, self.num_actions)

        # self.conv1 = nn.Conv3d(1, 128, 3, stride=1),
        # self.pool1 = nn.MaxPool3d(2),
        # self.conv2 = nn.Conv3d(128, 256, 3, stride=1),
        # self.conv3 = nn.Conv3d(256, 256, 2, stride=1),
        # self.relu = nn.ReLU(),
        # self.fc1 = nn.Linear(256, 512),
        # self.fc2 = nn.Linear(3, 512),
        # self.fc_out = nn.Linear(512, self.num_actions)
    
    def forward(self, s_input):
        (state, loc) = s_input
        state = state.unsqueeze(1)
        x_state_fe = self.state_cnn(state)
        x_state = self.state_model(x_state_fe.view(state.shape[0], -1))
        x_loc = self.loc_model(loc)
        x = x_state + x_loc
        out = self.fc_out(x)

        # x_state_fe = self.conv1(state),
        # x_state_fe = self.relu(x_state_fe),
        # x_state_fe = self.pool1(x_state_fe),
        # x_state_fe = self.conv2(x_state_fe),
        # x_state_fe = self.relu(x_state_fe),
        # x_state_fe = self.conv3(x_state_fe),
        # x_state_fe = self.relu(x_state_fe),
        # x_state = self.fc1(x_state_fe.view(state.shape[0], -1)),
        # x_state = self.relu(x_state),
        # x_loc = self.fc2(loc),
        # x_loc = self.relu(),
        # x = x_state + x_loc
        # out = self.fc_out(x)
        return out