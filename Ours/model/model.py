import torch
from torch import nn
import torch.nn.functional as F
from Ours.model.calss import CalSS
from torch_geometric.nn import DenseGCNConv


class GAN(nn.Module):
    def __init__(self, feat_dim, node_num, hidden_num=3, class_num=2, nheads=5):
        super(GAN, self).__init__()
        self.class_num = class_num
        assign_dim = node_num
        self.bn = True
        self.dropout = 0.4

        self.attentions1 = DenseGCNConv(feat_dim, hidden_num * nheads)
        self.out_att1 = DenseGCNConv(hidden_num * nheads, hidden_num * nheads)
        self.out_atta = DenseGCNConv(hidden_num * nheads, 20)
        self.linear11 = nn.Linear(1800, 256)
        self.linear12 = nn.Linear(256, self.class_num)

        self.attentions2 = DenseGCNConv(feat_dim, hidden_num * nheads)
        self.out_att2 = DenseGCNConv(hidden_num * nheads, hidden_num * nheads)
        self.out_attb = DenseGCNConv(hidden_num * nheads, 20)
        self.linear21 = nn.Linear(1800, 256)
        self.linear22 = nn.Linear(256, self.class_num)

        self.attentions3 = DenseGCNConv(feat_dim, hidden_num * nheads)
        self.out_att3 = DenseGCNConv(hidden_num * nheads, hidden_num * nheads)
        self.out_attc = DenseGCNConv(hidden_num * nheads, 20)
        self.linear31 = nn.Linear(1800, 256)
        self.linear32 = nn.Linear(256, self.class_num)

        self.linear4 = nn.Linear(40, 8)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(256)

    def forward(self, x1, adj1, x2, adj2, x3, adj3):

        # x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = F.elu(self.attentions1(x1, adj1))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        xa = self.linear4(x1)

        # x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = F.elu(self.attentions2(x2, adj2))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        xb = self.linear4(x2)

        # x3 = F.dropout(x3, self.dropout, training=self.training)
        x3 = F.elu(self.attentions3(x3, adj3))
        x3 = F.dropout(x3, self.dropout, training=self.training)
        xc = self.linear4(x3)

        xab = CalSS(xa, xb)
        xac = CalSS(xa, xc)
        xbc = CalSS(xb, xc)
        xa1 = torch.cat([xab, xac], dim=2)
        xa1 = F.softmax(xa1, dim=2)
        xb1 = torch.cat([xab, xbc], dim=2)
        xb1 = F.softmax(xb1, dim=2)
        xc1 = torch.cat([xbc, xac], dim=2)
        xc1 = F.softmax(xc1, dim=2)
        x1 = x1 + torch.mul(x2, torch.unsqueeze(xa1[:, :, 0], dim=2)) + torch.mul(x3, torch.unsqueeze(xa1[:, :, 1], dim=2))
        x2 = x2 + torch.mul(x1, torch.unsqueeze(xb1[:, :, 0], dim=2)) + torch.mul(x3, torch.unsqueeze(xb1[:, :, 1], dim=2))
        x3 = x3 + torch.mul(x2, torch.unsqueeze(xc1[:, :, 0], dim=2)) + torch.mul(x1, torch.unsqueeze(xc1[:, :, 1], dim=2))

        x1 = F.elu(self.out_att1(x1, adj1))
        x2 = F.elu(self.out_att2(x2, adj2))
        x3 = F.elu(self.out_att3(x3, adj3))

        x1 = self.out_atta(x1, adj1)
        x2 = self.out_attb(x2, adj2)
        x3 = self.out_attc(x3, adj3)

        out1 = x1.view(x1.size()[0], -1)
        out1 = self.linear11(out1)
        out1 = self.bn4(out1)
        out1 = F.elu(out1)
        ypred1 = self.linear12(out1)

        out2 = x2.view(x2.size()[0], -1)
        out2 = self.linear21(out2)
        out2 = self.bn5(out2)
        out2 = F.elu(out2)
        ypred2 = self.linear22(out2)

        out3 = x3.view(x3.size()[0], -1)
        out3 = self.linear31(out3)
        out3 = self.bn6(out3)
        out3 = F.elu(out3)
        ypred3 = self.linear32(out3)

        ypred = 1 * ypred1 + 1 * ypred2 + 1 * ypred3

        out = torch.cat([out1, out2, out3], dim=1)
        outt = F.normalize(out, p=2, dim=1)

        return ypred, outt
