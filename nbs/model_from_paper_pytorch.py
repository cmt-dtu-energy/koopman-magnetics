import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, reg_weight=0.0):
        super().__init__()
        # BN -> ReLU -> 1x1 conv
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        # BN -> ReLU -> 3x3 conv
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        # BN -> ReLU -> 1x1 conv
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out = F.relu(self.bn3(out))
        out = self.conv3(out)
        return out

class KoopmanModel(nn.Module):
    def __init__(self, num_filters, code_dim, seq_length, control_input=False, action_dim=0, halve_seq=False, recursive_pred=False, reg_weight=0.0, l2_regularizer=1e-6):
        super().__init__()
        self.seq_length = seq_length
        self.control_input = control_input
        self.halve_seq = halve_seq
        self.recursive_pred = recursive_pred
        
        # Encoder: downconv + bottleneck
        enc_layers = []
        in_ch = 4
        for out_ch in num_filters:
            enc_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False))
            enc_layers.append(BottleneckBlock(out_ch, out_ch//2, out_ch, reg_weight))
            in_ch = out_ch
        self.encoder_conv = nn.Sequential(*enc_layers)
        # to code
        self.fc_to_code = nn.Linear(num_filters[-1]* (128 // (2**len(num_filters))) * (256 // (2**len(num_filters))), code_dim)

        # Dynamics matrices A and optional B
        self.learn_A = False
        self.register_buffer('A', torch.zeros(code_dim, code_dim))  # will set later via least squares
        if control_input:
            self.B = nn.Parameter(torch.zeros(action_dim, code_dim))
        else:
            self.B = None
        
        # Decoder: reverse of encoder
        self.fc_from_code = nn.Linear(code_dim, num_filters[-1]* (128 // (2**len(num_filters))) * (256 // (2**len(num_filters))))
        dec_layers = []
        rev_filters = list(reversed(num_filters))
        in_ch = rev_filters[0]
        for out_ch in rev_filters[1:]:
            dec_layers.append(BottleneckBlock(in_ch, in_ch//2, in_ch, reg_weight))
            dec_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))
            in_ch = out_ch
        # final bottleneck + upconv back to 4 channels
        dec_layers.append(BottleneckBlock(in_ch, in_ch//2, in_ch, reg_weight))
        dec_layers.append(nn.ConvTranspose2d(in_ch, 4, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoder_conv = nn.Sequential(*dec_layers)

    def encode(self, x):
        # x: (batch*(seq+1),128,256,4) -> (batch*(seq+1),C,H,W)
        h = self.encoder_conv(x.permute(0,3,1,2))
        h = h.flatten(start_dim=1)
        code = self.fc_to_code(h)
        return code

    def solve_A(self, X, Y, lam=1e-1):
        # X, Y: (batch, seq, code_dim)
        hi = self.seq_length//2 if self.halve_seq else self.seq_length
        Xh = X[:,:hi,:].reshape(-1, X.size(-1))  # (n_samples, d)
        Yh = Y[:,:hi,:].reshape(-1, Y.size(-1))
        if self.B is not None:
            # subtract control effect
            U = self.u[:,:hi].reshape(-1, self.B.size(0))
            Yh = Yh - U @ self.B
        # least squares: A = (X^T X + lam I)^{-1} X^T Y
        XtX = Xh.t() @ Xh
        reg = lam * torch.eye(XtX.size(0), device=XtX.device)
        A = torch.linalg.solve(XtX + reg, Xh.t() @ Yh)
        self.A.copy_(A)

    def forward(self, x, u=None):
        # x: (batch, seq+1, 128,256,4)
        b, s1, H, W, C = x.shape
        x_ = x.reshape(b*(s1), H, W, C)
        codes = self.encode(x_)  # (b*(s1), code_dim)
        codes = codes.view(b, s1, -1)
        code_x = codes[:,:self.seq_length,:]
        code_y = codes[:,1:,:]
        if self.control_input:
            self.u = u  # store for solve_A
        self.solve_A(code_x, code_y)
        # one-step prediction
        y_pred = code_x @ self.A.t()
        if self.control_input:
            y_pred = y_pred + u @ self.B
        # recursive prediction
        if self.recursive_pred:
            y_list = [y_pred[:,0:1,:]]
            for t in range(1, self.seq_length):
                y_t = y_list[-1] @ self.A.t()
                if self.control_input:
                    y_t = y_t + u[:,t:t+1,:] @ self.B
                y_list.append(y_t)
            y_pred = torch.cat(y_list, dim=1)
        # assemble codes for decoding
        cat_codes = torch.cat([code_x.reshape(-1, codes.size(-1)), y_pred.reshape(-1, codes.size(-1))], dim=0)
        # decode
        h_dec = self.fc_from_code(cat_codes)
        H2 = H // (2**len(self.encoder_conv))
        W2 = W // (2**len(self.encoder_conv))
        h_dec = h_dec.view(-1, self.encoder_conv[-1].out_channels, H2, W2)
        rec = self.decoder_conv(h_dec)
        # split reconstruction for x and y
        rec = rec.permute(0,2,3,1)
        rec_x = rec[:b*self.seq_length]
        rec_y = rec[b*self.seq_length:]
        return rec_x, rec_y

# Example optimizer and loss setup:
# model = KoopmanModel(num_filters=[16,32,64], code_dim=128, seq_length=10, control_input=True, action_dim=2)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.MSELoss()
