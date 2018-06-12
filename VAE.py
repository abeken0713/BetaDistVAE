import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Assume image reconstruction

def log_clip(x, xmin=1e-10):
    return torch.log(torch.clamp(x, min=xmin))

class VAE(nn.Module):
    def __init__(self, in_dim, mid_dim, hid_dim):
        super(VAE, self).__init__()
        self.__name__ = "VAE"

        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.hid_dim = hid_dim

        self.enc = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 2*hid_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(hid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, n_dim)
        msr = self.enc(x)
        mr, sr_ = torch.chunk(msr, 2, 1)
        sr = F.softplus(sr_)
        if self.training:
            with torch.no_grad():
                eps = torch.randn_like(mr)
            z = mr + eps*sr
        else:
            z = mr
        mg = self.dec(z)
        return mr, sr, mg

    def loss(self, x):
        # x: (batch_size, n_dim)
        mr, sr, mg = self.forward(x)
        # mr, sr: (batch_size, hid_dim)
        # mg: (batch_size, n_dim)
        nll = self._bernoulli_nll(x, mg)    # (batch_size,)
        kld = self._gaussian_kld(mr, sr)    # (batch_size,)
        out = nll + kld
        return torch.mean(out)

    def _bernoulli_nll(self, x, m):
        # x, m: (batch_size, n_dim)
        out = x*log_clip(m) + (1 - x)*log_clip(1 - m)
        return -out.sum(1)

    def _gaussian_kld(self, m, s):
        # m, s: (batch_size, hid_dim)
        tmp = 1 + 2*torch.log(s) - m*m - s*s
        return -0.5*tmp.sum(1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from MachineLearning.LoadData import load_mnist
    from MachineLearning.utils import splitbatch
    from torch import optim
    # ----- Parameters -----
    n_epoch = 20
    batch_size = 100
    log_step = 1
    n_valid = 5000
    n_row = 3
    n_col = 3
    n_test = n_row*n_col
    # ----------------------
    X_train, X_test, y_train, y_test = load_mnist()
    idx_valid = np.random.permutation(y_train.shape[0])
    X_valid, X_train = np.split(X_train[idx_valid], (n_valid,))
    X_valid, X_train, X_test = torch.Tensor(X_valid)/255, torch.Tensor(X_train)/255, torch.Tensor(X_test)/255
    model = VAE(784, 100, 10)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(n_epoch):
        total_cost = 0.
        for n, (X_batch,) in enumerate(splitbatch([X_train], batch_size, shuffle=True)):
            model.train()
            optimizer.zero_grad()
            loss = model.loss(X_batch)
            loss.backward()
            optimizer.step()
            total_cost += loss.item()
        if (epoch % log_step == -1 % log_step):
            total_cost /= (n + 1)
            # with torch.no_grad():
            model.eval()
            valid_cost = model.loss(X_valid).item()
            print("[{:03d}/{:03d}] loss/train: {:.2f}, loss/valid: {:.2f}".format(epoch + 1, n_epoch, total_cost, valid_cost))
    
    with torch.no_grad():
        fig = plt.figure(figsize=(8, 4))
        rng = np.random.RandomState(1234)
        X_test = X_test[rng.choice(y_test.shape[0], n_test, replace=False)]
        model.eval()
        _, _, X_pred = model(X_test)
        X_test = X_test.numpy()
        X_pred = X_pred.numpy()
        for i in range(n_row):
            for j in range(n_col):
                n = i*n_col + j
                t = X_test[n].reshape(28, 28)
                y = X_pred[n].reshape(28, 28)
                img = np.hstack([t, y])
                ax = fig.add_subplot(n_row, n_col, n+1)
                ax.imshow(img, "Greys")
                ax.set_xticks([])
                ax.set_yticks([])
                # ax = fig.add_subplot(n_row, 2*n_col, 2*n+1)
                # ax.imshow(t, "Greys")
                # ax.set_xticks([])
                # ax.set_yticks([])
                # ax = fig.add_subplot(n_row, 2*n_col, 2*n+2)
                # ax.imshow(y, "Greys")
                # ax.set_xticks([])
                # ax.set_yticks([])
        plt.show()