import torch
from torch import nn
from models.Identity import Identity


# follow the notation of 'non-local neural networks'
# https://arxiv.org/pdf/1711.07971v3.pdf
# version 1: Qx, Kaux, Vx
class AttentionV1(nn.Module):
    def __init__(self, Cin, Cout=None, batch_norm=False, zero_z=False, freeze_z=False):
        super(AttentionV1, self).__init__()
        if Cout is None:
            Cout = Cin
        self.Wq = nn.Sequential(nn.Conv2d(Cin, Cout // 2, kernel_size=1, padding=0), nn.BatchNorm2d(Cout // 2)) if batch_norm else nn.Conv2d(Cin, Cout // 2, kernel_size=1, padding=0)
        self.Wk = nn.Sequential(nn.Conv2d(Cin, Cout // 2, kernel_size=1, padding=0), nn.BatchNorm2d(Cout // 2)) if batch_norm else nn.Conv2d(Cin, Cout // 2, kernel_size=1, padding=0)
        self.Wv = nn.Sequential(nn.Conv2d(Cin, Cout // 2, kernel_size=1, padding=0), nn.BatchNorm2d(Cout // 2)) if batch_norm else nn.Conv2d(Cin, Cout // 2, kernel_size=1, padding=0)
        self.softmax = nn.Softmax(2)  # the batched column

        self.Wz = nn.Sequential(nn.Conv2d(Cout // 2, Cout, kernel_size=1, padding=0), nn.BatchNorm2d(Cout)) \
            if batch_norm else nn.Conv2d(Cout // 2, Cout, kernel_size=1, padding=0)

        # without breaking pretrianed behavior
        self.zero_z = zero_z
        if self.zero_z:
            self.Wz[0].weight.data.fill_(0) if batch_norm else self.Wz.weight.data.fill_(0)
            self.Wz[0].bias.data.fill_(0) if batch_norm else self.Wz.bias.data.fill_(0)

        # don't train !
        self.freeze_z = freeze_z
        if self.freeze_z:
            if batch_norm:
                self.Wz[0].weight.requires_grad = False
                self.Wz[0].bias.requires_grad = False
            else:
                self.Wz.weight.requires_grad = False
                self.Wz.bias.requires_grad = False

    def forward(self, x, aux):
        # assert len(x.shape) == 4  # ensure shape like B,C,H,W
        # assert x.shape == q.shape  # same shape

        B, C, H, W = x.shape

        # extract q k v
        bq, bk, bv = self.Wq(x), self.Wk(aux), self.Wv(x)  # trick: can subsample k and v simultaneously in spatial dim

        # reshape
        r = lambda x: x.view(B, C//2, -1)
        bq, bk, bv = r(bq), r(bk), r(bv)  # shape(B, C/2, Spatial), (B, C/2, Spatial'), (B, C/2, Spatial')

        # calculate attention
        KTQ = torch.bmm(bk.permute([0,2,1]), bq)  # KTQ means (K^T • Q) with shape(B, Spatial', Spatial)
        atten_map = self.softmax(KTQ)  # y = V • Softmax_column-wise(K^T • Q)
        y = torch.bmm(bv,atten_map)  # shape(B, C/2, Spatial)
        y = y.view(B,C//2,H,W)

        z = self.Wz(y) + x
        return z


# version 2:
# Qx, Kaux, Vaux
# batch norm only at Wz
# x resizable
class AttentionV2(nn.Module):
    def __init__(self, Cin, Cout=None, batch_norm=False, zero_z=False, freeze_z=False, Wx_enabled=False):
        super(AttentionV2, self).__init__()
        if Cout is None:
            Cout = Cin
        if not Wx_enabled:
            assert Cin == Cout

        self.Cin = Cin
        self.Cout = Cout

        self.Wq = nn.Conv2d(Cin, Cout // 2, kernel_size=1, padding=0)
        self.Wk = nn.Conv2d(Cin, Cout // 2, kernel_size=1, padding=0)
        self.Wv = nn.Conv2d(Cin, Cout // 2, kernel_size=1, padding=0)

        self.softmax = nn.Softmax(2)  # the batched column

        self.Wz = nn.Sequential(
            nn.Conv2d(Cout // 2, Cout, kernel_size=1, padding=0), nn.BatchNorm2d(Cout)
        ) \
            if batch_norm else nn.Conv2d(Cout // 2, Cout, kernel_size=1, padding=0)

        self.Wx = nn.Conv2d(Cin, Cout, kernel_size=1, padding=0)
        if not Wx_enabled:
            self.Wx = Identity()

        # without breaking pretrianed behavior
        self.zero_z = zero_z
        if self.zero_z:
            self.Wz[0].weight.data.fill_(0) if batch_norm else self.Wz.weight.data.fill_(0)
            self.Wz[0].bias.data.fill_(0) if batch_norm else self.Wz.bias.data.fill_(0)

        # don't train !
        self.freeze_z = freeze_z
        if self.freeze_z:
            if batch_norm:
                self.Wz[0].weight.requires_grad = False
                self.Wz[0].bias.requires_grad = False
            else:
                self.Wz.weight.requires_grad = False
                self.Wz.bias.requires_grad = False

    def forward(self, x, aux):
        # assert len(x.shape) == 4  # ensure shape like B,C,H,W
        # assert x.shape == q.shape  # same shape

        B, _, H, W = x.shape

        # extract q k v, b means batched
        bq, bk, bv = self.Wq(x), self.Wk(aux), self.Wv(aux)  # trick: can subsample k and v simultaneously in spatial dim

        # reshape
        r = lambda x: x.view(B, -1, H*W)
        bq, bk, bv = r(bq), r(bk), r(bv)  # shape(B, C/2, Spatial), (B, C/2, Spatial'), (B, C/2, Spatial')

        # calculate attention
        KTQ = torch.bmm(bk.permute([0,2,1]), bq)  # KTQ means (K^T • Q) with shape(B, Spatial', Spatial)
        atten_map = self.softmax(KTQ)  # y = V • Softmax_column-wise(K^T • Q)
        y = torch.bmm(bv,atten_map)  # shape(B, C/2, Spatial)
        y = y.view(B, -1, H, W)

        z = self.Wz(y) + self.Wx(x)
        return z


class MultiHead(nn.Module):
    def __init__(self, Cin, num_head=8, batch_norm=False, zero_z=False, freeze_z=False):
        super(MultiHead, self).__init__()
        self.Cin = Cin
        self.Cout = Cin
        self.out_channels = [self.Cin//num_head] * num_head  # list expension
        self.out_channels[-1] += self.Cin % num_head  # fill the last one
        self.heads = nn.ModuleList(
            (
                AttentionV2(self.Cin, cout,Wx_enabled=True, batch_norm=False, zero_z=zero_z, freeze_z=freeze_z)
                for cout in self.out_channels
            )
        )
        self.Wo = nn.Sequential(
            nn.Conv2d(self.Cout, self.Cin, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.Cout)
        ) \
            if batch_norm else nn.Conv2d(self.Cout, self.Cin, kernel_size=1, stride=1, padding=0)

        for param in self.Wo.parameters():
            if zero_z:
                param.data.fill_(0)
            if freeze_z:
                param.requires_grad = False

    def forward(self, x, aux):
        # B, C, H, W = x.shape
        X = [
            atten(x, aux)
            for atten in self.heads
        ]
        y = torch.cat(X, 1)
        z = self.Wo(y) + x
        return z


if __name__ == '__main__':
    mh = MultiHead(10,3)
    m = mh(torch.rand(1,10,5,5), torch.rand(1,10,5,5))
    pass
