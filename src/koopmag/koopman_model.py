import torch
from torch import nn


class BottleNeck(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, act_fn=nn.Tanh) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            act_fn(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            act_fn(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_fn(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x) -> torch.Tensor:
        return self.conv(x) + x
    

class KoopmanAE(nn.Module):

    def __init__(self, chns, latent_dim, act_fn=nn.Tanh) -> None:
        super().__init__()
        self.chns = chns
        self.act_fn = act_fn
        self.latent_dim = latent_dim

        self.flat = nn.Flatten()

        self._init_encoder()
        self.fc_encoder = nn.Linear(chns[-1] * (40 // 2**len(chns)) * (16 // 2**len(chns)), latent_dim)

        self.fc_decoder = nn.Linear(latent_dim, chns[-1] * (40 // 2**len(chns)) * (16 // 2**len(chns)))
        self._init_decoder()
        self.tanh = nn.Tanh()

    def _init_encoder(self) -> None:

        encoder_blocks = []
        in_ch = 3
        for ch in self.chns:
            encoder_blocks.append(nn.Conv2d(in_ch, ch, kernel_size=3, padding=1, stride=2, bias=False))
            encoder_blocks.append(BottleNeck(ch, ch // 2, ch, act_fn=self.act_fn))
            in_ch = ch
        
        self.encoder_block = nn.Sequential(*encoder_blocks)
    
    def _init_decoder(self) -> None:

        decoder_blocks = []
        reversed_chns = self.chns[::-1]
        in_ch = reversed_chns[0]
        for ch in reversed_chns[1:]:
            decoder_blocks.append(BottleNeck(in_ch, in_ch // 2, in_ch, act_fn=self.act_fn))
            decoder_blocks.append(nn.ConvTranspose2d(in_ch, ch, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False))
            in_ch = ch

        decoder_blocks.append(BottleNeck(in_ch, in_ch // 2, in_ch, act_fn=self.act_fn))
        decoder_blocks.append(nn.ConvTranspose2d(in_ch, 3, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False))

        self.decoder_block = nn.Sequential(*decoder_blocks)

    def encode(self, x) -> torch.Tensor:
        x = self.encoder_block(x)    # encode X: (n x m)  -> Xtilde: (n x d)
        x = self.flat(x)             # flatten Xtilde: (n x d)
        x = self.fc_encoder(x)
        return x
    
    def decode(self, x) -> torch.Tensor:
        x = self.fc_decoder(x)
        x = x.view(-1, self.chns[-1], 40 // 2**len(self.chns), 16 // 2**len(self.chns))
        x = self.decoder_block(x)    # decode Xtilde: (n x d)  -> X: (n x m)
        x = self.tanh(x)            # apply activation function
        return x
    
    def forward(self, X) -> torch.Tensor:
        X = X.permute(0, 1, 4, 2, 3)    # (batch_size, nseq, n_x, n_y, 3) -> (batch_size, nseq, 3, n_x, n_y)
        batch_size, nseq, nchan, n_x, n_y = X.shape
        X = X.view(-1, nchan, n_x, n_y)
        x = self.encode(X)    # encode X: (n x m)  -> Xtilde: (n x d)
        x = self.decode(x)
        x = x.view(batch_size, nseq, nchan, n_x, n_y)    # decode Xtilde: (n x d)  -> X: (n x m)
        x = x.permute(0, 1, 3, 4, 2)    # (batch_size, nseq, n_x, n_y, 3)
        return x
    


class DeepKoopman(nn.Module):

    def __init__(self, chns, latent_dim, act_fn=nn.Tanh, learn_A=False, method="pinv", lam=1e-1) -> None:
        super().__init__()
        
        self.autoencoder = KoopmanAE(chns, latent_dim, act_fn=act_fn)
        self.latent_dim = latent_dim
        self.learn_A = learn_A
        self.method = method
        self.lam = lam

        if learn_A:
            self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False) # shape (d, d)
        else:
            self.register_buffer("A", torch.empty(self.latent_dim, self.latent_dim)) # shape (d, d)

        self.B = nn.Linear(3, self.latent_dim, bias=False) # shape (n_magnetization, d)


    def compute_linear_operator_pinv(self, Xtilde, Ytilde) -> None:
        A = torch.linalg.pinv(Xtilde, rcond=1e-2) @ Ytilde
        self.A.data.copy_(A)
    

    def compute_linear_operator_tikh(self, Xtilde, Ytilde) -> None:
    # Xtilde: (n, d), Ytilde: (n, d)
        XtX = Xtilde.T @ Xtilde             # (d, d)
        d   = XtX.shape[0]
        reg = self.lam * torch.eye(d, device=XtX.device)
        A   = torch.linalg.solve(XtX + reg, Xtilde.T @ Ytilde)
        self.A.data.copy_(A)
        

    def compute_linear_operator(self, Xtilde, Ytilde) -> None:
        
        # flatten from (batch_size, nseq, d) to (batch_size * seq, d)
        b, nseq, d = Xtilde.shape
        xtilde = Xtilde.reshape(b * nseq, d)
        ytilde = Ytilde.view(b * nseq, d) 

        if self.method == "pinv":
            self.compute_linear_operator_pinv(xtilde, ytilde)
        elif self.method == "tikh":
            self.compute_linear_operator_tikh(xtilde, ytilde)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'pinv' or 'tikh'.")
        
        if torch.any(torch.isnan(self.A.data)):
            raise ValueError("nan values detected in A matrix")


    def forward(self, X, U) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        params:
        X: input data (n x m)
        Y: target data (n x m)
        U: external effect (n x 3)
        '''
        X = X.permute(0, 1, 4, 2, 3)  # switch data dimensions to match pytorch conv layer
        batch_size, nseq, nchan, n_x, n_y = X.shape     # extract dimensions
        T_half = nseq // 2

        X = X.view(batch_size*nseq, nchan, n_x, n_y)  # reshape to (batch_size*nseq, nchan, n_x, n_y)

        # encode data 
        X = self.autoencoder.encode(X)    # encode X: (batch_size*nseq, nchan, n_x, n_y)  -> Xtilde: (batch_size*nseq, d)

        # reshape back to (batch_size, nseq, ...)
        X = X.view(batch_size, nseq, -1)  # reshape to (batch_size, nseq, d)

        Xtilde = X[:, :-1, :]       # X data
        Ytilde = X[:, 1:, :]        # Y data

        # apply B to U:
        U = U.view(batch_size * nseq, -1)  # reshape to (batch_size, nseq, nchan)
        Bu = self.B(U)
        Bu = Bu.view(batch_size, nseq, -1)  # reshape to (batch_size, nseq, d)

        ext_effect = self.B(U)  # shape (n//2 x d)
        ext_effect = ext_effect.view(batch_size, nseq, -1)  # reshape to (batch_size, nseq, d)

        # extract first half of external effect
        ext_effect = ext_effect[:, :T_half, :]  # take first half of the sequence

        # compute recursive predictions
        if not self.learn_A:

            # compute linear operator using pseudo-inverse or tikhonov regularization
            self.compute_linear_operator(
                Xtilde[:, :T_half,:], Ytilde[:, :T_half,:] - ext_effect
            )

            # compute predictions
            Ytilde_pred_list = [Xtilde[:, 0, :].matmul(self.A) + Bu[:, 0, :]] 
            for t in range(1, nseq-1):
                next_pred = Ytilde_pred_list[t-1] @ self.A + Bu[:, t, :]
                Ytilde_pred_list.append(next_pred)

        # If A is learned as a parameter:
        else:
            Ytilde_pred_list = [self.A(Xtilde[:, 0, :]) + Bu[:, 0, :]] 
            for t in range(1, nseq-1):
                next_pred = self.A(Ytilde_pred_list[-1]) + Bu[:, t, :]
                Ytilde_pred_list.append(next_pred)
        
        Ytilde_pred = torch.stack(Ytilde_pred_list, dim=1)

        # concatenate Xtilde and Ytilde_pred for efficient decoding
        cat_data = torch.cat((Xtilde.reshape(-1, self.latent_dim), Ytilde_pred.reshape(-1, self.latent_dim)), dim=0)

        decoded = self.autoencoder.decode(cat_data)

        Xhat = decoded[:batch_size*(nseq-1), :, :, :]  # shape (batch_size*nseq, n_x, n_y, 3)
        Xhat = Xhat.view(batch_size, nseq-1, nchan, n_x, n_y)  # reshape to (batch_size, nseq-1, n_x, n_y, 3)

        Xhat = Xhat.permute(0, 1, 3, 4, 2)  # switch data dimensions back to (n_seq, n_tsteps - 1, n_x, n_y, 3)

        Yhat = decoded[batch_size*(nseq-1):, :, :, :]  # shape (batch_size*nseq, n_x, n_y, 3)
        Yhat = Yhat.view(batch_size, nseq-1, nchan, n_x, n_y)

        Yhat = Yhat.permute(0, 1, 3, 4, 2)  # switch data dimensions back to (n_seq, n_tsteps - 1, n_x, n_y, 3)

        return Xhat, Yhat
    
    
    def predict(self, X, U) -> torch.Tensor:
        '''
        params:
        X: first observation (n_batch, n_x, n_y, 3)
        T: number of time steps to predict
        U: external effect (n_batch, T, 3)
        '''
    
        assert len(X.shape) == 4, f"X should be of shape (n_batch, n_x, n_y, 3), but got {X.shape}"

        self.eval()

        T = U.shape[1]

        with torch.no_grad():
            n_batch, n_x, n_y, n_chan = X.shape
            X = X.permute(0, 3, 1, 2)  # switch data dimensions to match pytorch conv layer
            Xtilde0 = self.autoencoder.encode(X)    # encode X: (n_batch, n_chan, n_x, n_y)  -> Xtilde: (n_batch x latent_dim)

            U = U.view(n_batch * T, -1)  # reshape to (batch_size, nseq, nchan)
            Bu = self.B(U)
            Bu = Bu.view(n_batch, T, -1)  # reshape to (batch_size, nseq, d)
            # If A is computed using pseudo-inverse or tikhonov regularization:

            Ytilde_pred = [Xtilde0]
            if not self.learn_A:
                for t in range(T):
                    Ytilde_pred.append(Ytilde_pred[-1].matmul(self.A) + Bu[:, t, :])

            # If A is learned as a parameter:
            else:
                for t in range(T):
                    next_pred = self.A(Ytilde_pred[-1]) + Bu[:, t, :]
                    Ytilde_pred.append(next_pred)

            # concatenate along dim 1 (time dimension)
            Ytilde_pred = torch.stack(Ytilde_pred, dim=1)  # -> (n_batch, T+1, latent_dim)

            # reshape for decoding
            Ytilde_pred = Ytilde_pred.view(n_batch * (T + 1), -1) # (n_batch, T+1, latent_dim) -> (n_batch*(T+1), latent_dim)
            Yhat = self.autoencoder.decode(Ytilde_pred)
            Yhat = Yhat.view(n_batch, T+1, n_chan, n_x, n_y)
            Yhat = Yhat.permute(0, 1, 3, 4, 2)
        return Yhat