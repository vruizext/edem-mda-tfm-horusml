from fastai.vision.all import *


class RegressionResnet(Module):
    def __init__(self, model_arch):
        model = model_arch(pretrained=True)
        self.encoder = TimeDistributed(create_body(model))
        self.head = TimeDistributed(create_head(model.fc.in_features, 1))

    def forward(self, x):
        x = torch.stack(x, dim=1)
        return self.head(self.encoder(x)).mean(dim=1)


class RegressionResnetLSTM(Module):
    def __init__(self, model_arch, hidden_size=512, num_layers=4, dropout=0.5):
        self.resnet = model_arch(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1], nn.Flatten())
        self.encoder = TimeDistributed(self.resnet)
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_layers * hidden_size, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.stack(x, dim=1)
        enc_out = self.encoder(x)
        lstm_out, (hidden, _) = self.lstm(enc_out)
        batch_size = x.shape[0]
        out = hidden.permute(1, 0, 2).contiguous().view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.mish(out)
        out = self.fc2(out)
        return out


def resnet_splitter(model):
    return [params(model.encoder), params(model.head)]


def resnet_lstm_splitter(model):
    return [params(model.encoder), params(model.lstm) + params(model.fc1) + params(model.fc2)]

