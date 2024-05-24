---
layout: post
title: To study Machine Learning/VAE
date: 2024-05-24 11:14 +0800
last_modified_at: 2024-01-24 16:08:25 +0800
tags: [jekyll theme, jekyll, tutorial]
toc:  true
---

## VAE(Variational Autoencoder)
 잠재 벡터 표현할 때에는 잠재 벡터를 원본 이미지로 다시 디코딩할 수 있어야하는데, 기존의 autoencoder은 분리되고 비연속적인 잠재 공간 $Z$가 발생할 수 있다는 제약이 존재했다. 
 > VAE는 이러한 한계를 극복하기 위하여 제시된 방법이며 input image X를 잘 설명하는 feature를 추출하여 Latent vector z에 담고, 이 Latent vector z를 통해 X와 유사하지만 완전히 새로운 데이터를 생성해내는 것을 목표로 한다. 
<br><br>

VAE는 input를 잠재 벡터에 대한 확률 분포에 매핑한 다음, 잠재 벡터를 샘플링한다는 점에서 기존의 autoencoder과 다르다. 이는 더 나은 decoder를 제공할 수 있게 해준다.
<br><br>
특히, VAE는 input $x$를 잠재 벡터 $z=e(x)$에 매핑하는 대신 평균 벡터 $\mu_{z}(x)$ 와 대각화 가우스 분포 $\mathscr{N}({\mu}_{z},{\sigma}^2_{z})$ 를 매개변수화하는 분산 벡터 ${\sigma}^2_{z}(x)$ 에 매핑한다. 그러면 잠재 벡터 $z$ 는 $z \sim \mathscr{N}(\mu_z,{\sigma}^2_{z})$에 샘플링되게 된다.
<br><br>
아래부터는 VAE를 사용하여 MNIST 데이터 셋의 이미지를 생성하고 재구성하는 예제를 구현해볼 것이다.

1. 기본 설정
- 필요한 라이브러리와 장치 설정
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions as D
import torchvision
import numpy as np
import matplotlib.pyplot as plt
```
<br/>

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
```
<br/>

- MNIST 데이터 로드
```python
batchsize = 128
nplot_test = 5
dim_x_list = [28, 28]
dim_x = dim_x_list[0] * dim_x_list[1]
traindata = torchvision.datasets.MNIST('./traindata',
                                       transform=torchvision.transforms.ToTensor(),
                                       train=True, download=True)
testdata = torchvision.datasets.MNIST('./testdata',
                                      transform=torchvision.transforms.ToTensor(),
                                      train=False, download=True)
traindataloader = torch.utils.data.DataLoader(traindata, batch_size=batchsize, shuffle=True)
testdataloader = torch.utils.data.DataLoader(testdata, batch_size=nplot_test, shuffle=True)

```


2. VAE 모델 정의
- VAE  클래스 정의
```python
class VAE(nn.Module):
  def __init__(self, **kwargs):
    super(VAE, self).__init__()
    self.__dict__.update(kwargs)

    # Encoder
    self.dense1_enc = nn.Linear(self.dim_x, self.dim_h)
    self.dense2_enc = nn.Linear(self.dim_h, self.dim_h)
    self.dense_mu_enc = nn.Linear(self.dim_h, self.dim_z)
    self.dense_logvar_enc = nn.Linear(self.dim_h, self.dim_z)

    # Decoder
    self.dense1_dec = nn.Linear(self.dim_z, self.dim_h)
    self.dense2_dec = nn.Linear(self.dim_h, self.dim_h)
    self.dense3_dec = nn.Linear(self.dim_h, self.dim_x)

    self.relu = nn.ReLU()

  def encode(self, x):
    x = torch.flatten(x, start_dim=1)
    _h1 = self.relu(self.dense1_enc(x))
    _h2 = self.relu(self.dense2_enc(_h1))
    mu = self.dense_mu_enc(_h2)
    logvar = self.dense_logvar_enc(_h2)
    return mu, logvar

  def decode(self, z):
    _h1 = self.relu(self.dense1_dec(z))
    _h2 = self.relu(self.dense2_dec(_h1))
    xhat = torch.sigmoid(self.dense3_dec(_h2))
    return xhat.reshape((-1, 1, self.dim_x_list[0], self.dim_x_list[1]))

  def reparam(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + std * eps
    return z

  def forward(self, x):
    x = torch.flatten(x, start_dim=1)
    mu, logvar = self.encode(x)
    z = self.reparam(mu, logvar)
    xhat = self.decode(z)
    return xhat, mu, logvar

  @staticmethod
  def loss_function(x, xhat, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(xhat, x, reduction='sum')
    p = D.Normal(torch.zeros_like(mu), torch.ones_like(logvar))
    q = D.Normal(mu, logvar.exp())
    _kld = D.kl.kl_divergence(q, p)
    reg_loss = torch.sum(torch.clamp(_kld, min=0.0))
    return recon_loss, reg_loss

```
<br/>

3. 모델 학습 및 평가
- 학습 및 손실 계산
```python
dim_z = 2
dim_h = 256
epochs = 30
displaystep = 5
plotheight = 1.33
vae = VAE(dim_x=dim_x, dim_x_list=dim_x_list, dim_h=dim_h, dim_z=dim_z).to(device)

opt = torch.optim.Adam(vae.parameters(), lr=1e-4)

reconloss_list, regloss_list = [], []
for epoch in range(epochs):
  reconloss, regloss = [], []
  vae.train()
  for x, y in traindataloader:
    x = x.float().to(device)
    opt.zero_grad()
    _x_hat, _mu, _logvar = vae(x)
    _recon_loss, _reg_loss = vae.loss_function(x, _x_hat, _mu, _logvar)
    _vae_loss = _recon_loss + _reg_loss
    _vae_loss.backward()
    opt.step()
    reconloss.append(_recon_loss.item())
    regloss.append(_reg_loss.item())

  _recon_l_mean, _recon_l_std = np.mean(reconloss), np.std(reconloss)
  reconloss_list.append([_recon_l_mean, _recon_l_std])
  _reg_l_mean, _reg_l_std = np.mean(regloss), np.std(regloss)
  regloss_list.append([_reg_l_mean, _reg_l_std])

  if (epoch + 1) % displaystep == 0 or (epoch + 1) in [1]:
    print("{:4d} loss(recon): {:7.5f}/{:7.5f}, loss(reg): {:7.5f}/{:7.5f}".
          format(epoch + 1, _recon_l_mean, _recon_l_std, _reg_l_mean, _reg_l_std))
    draw_procedure(vae, testdataloader, nplot_test, plotheight)

```
<br/>
- 손실 함수 시각화

```python
reconloss_list, regloss_list = np.array(reconloss_list), np.array(regloss_list)
plot_xrange = np.arange(0, reconloss_list.shape[0])

vae.eval()
fig = plt.figure(figsize=(9, 2))
reconloss_mean = np.log(reconloss_list[:, 0])
reconloss_upper = np.log(reconloss_list[:, 0] + reconloss_list[:, 1])
reconloss_lower = np.log(reconloss_list[:, 0] - reconloss_list[:, 1])
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(plot_xrange, reconloss_mean, 'b--', linewidth=1, label="recon-loss")
ax1.fill_between(plot_xrange, reconloss_upper, reconloss_lower, color='k', alpha=0.2)
ax1.grid()
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss(log-scale)")
ax1.set_title("reconstruction-loss")

regloss_mean = np.log(regloss_list[:, 0])
regloss_upper = np.log(regloss_list[:, 0] + regloss_list[:, 1])
regloss_lower = np.log(regloss_list[:, 0] - regloss_list[:, 1])
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(plot_xrange, regloss_mean, 'm--', linewidth=1, label="reg-loss")
ax2.fill_between(plot_xrange, regloss_upper, regloss_lower, color='k', alpha=0.2)
ax2.grid()
ax2.set_xlabel("epoch")
ax2.set_ylabel("loss(log-scale)")
ax2.set_title("regularization-loss")
plt.show()
```
4. 잠재 공간 시각화
- 잠재 공간 시각화 함수
```python
def plot_latent(model, data, num_batches=100):
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  fig = plt.figure(figsize=(7, 7))
  ax = fig.add_subplot(1, 1, 1)
  ax.set_aspect("equal")
  for i, (x, y) in enumerate(data):
    x = x.float().to(device)
    mu, logvar = model.encode(x)
    mu = mu.to('cpu').detach().numpy()
    _s = ax.scatter(mu[:, 0], mu[:, 1], c=y, alpha=0.5, cmap='tab10')
    if i > num_batches:
      divider = make_axes_locatable(ax)
      cax = divider.append_axes("right", size="5%", pad=0.05)
      plt.colorbar(_s, cax=cax)
      break
  ax.set_xlabel("$\mu_{z}(1)$")
  ax.set_ylabel("$\mu_{z}(2)$")
  ax.grid()
  plt.show()
  plt.close()

plot_latent(vae, traindataloader)
```
<br/>