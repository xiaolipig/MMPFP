import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



class RepVGGBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, deploy=False):

        super(RepVGGBlock1D, self).__init__()
        self.deploy = deploy
        kernel_size = 3
       
        padding = 0

        if deploy:
            self.rbr_reparam = nn.Conv1d(in_channels, out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=True)
        else:
            
            self.rbr_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm1d(in_channels)
            else:
                self.rbr_identity = None
        self.nonlinearity = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.nonlinearity(self.rbr_reparam(x))
        out = self.rbr_conv(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out += self.rbr_identity(x)
        return self.nonlinearity(out)



class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=256, num_head=4):
        super(CNN, self).__init__()
        self.num_head = num_head

        self.one_hot_embed = nn.Embedding(21, 96)
        self.proj_aa = nn.Linear(96, 512)
        self.proj_esm = nn.Linear(1280, 512)
        self.emb = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=0),
            nn.BatchNorm1d(hidden_dim)
        )
        self.ms_f = FPN_Module(hidden_dim)
        self.multi_head = MCAM(self.num_head, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x_aa = self.one_hot_embed(data.native_x.long())  # [N, 96]
        x_aa = self.proj_aa(x_aa)
        x = data.x.float()
        x_esm = self.proj_esm(x)
        x = F.relu(x_aa + x_esm)
        batch_x, _ = to_dense_batch(x, data.batch)
        x = batch_x.permute(0, 2, 1)
        conv_emb = self.emb(x)
        conv_ms = self.ms_f(conv_emb)
        conv_x = self.multi_head(conv_emb)
        conv_mha = self.multi_head(conv_ms)
        out = conv_x + conv_mha
        output = torch.flatten(out, 1)
        output = self.fc_out(output)
        output = torch.sigmoid(output)
        return output




class FPN(nn.Module):
    def __init__(self, hidden_dim):
        super(FPN, self).__init__()
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class MCAM(nn.Module):
    def __init__(self, num_heads, hidden_dim):
        super(MCAM, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        pooled = self.pool(x)  # [B, hidden_dim, 1]
        return pooled.squeeze(-1)  # [B, hidden_dim]



if __name__ == "__main__":

    class DummyData:
        pass


    data = DummyData()
    N = 100
    data.native_x = torch.randint(0, 21, (N,))
    data.x = torch.randn(N, 1280)
    data.batch = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])

    model = CNN(input_dim=512, hidden_dim=64, num_classes=256, num_head=4)
    out = model(data)
    print(out.shape)
