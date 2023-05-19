from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import save_image

dataset = CIFAR10(root='./data', download=True,
                  transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
batch_size = 64
for batch_idx, data in enumerate(dataloader):
    if batch_idx == len(dataloader) - 1:
        continue
    real_images, _ = data

    print('#{} has {} images.'.format(batch_idx, batch_size))
    if batch_idx % 100 == 0:
        path = './test_data1/CIFAR10_shuffled_batch{:03d}.png'.format(batch_idx)
        save_image(real_images, path, normalize=True)
import torch.nn as nn
latent_size = 64
n_channel = 3
n_g_feature = 64
"生成网络采用了四层转置卷积操作"
gnet = nn.Sequential(

    nn.ConvTranspose2d(latent_size, 4 * n_g_feature, kernel_size=4,
                       bias=False),
    nn.BatchNorm2d(4 * n_g_feature),
    nn.ReLU(),

    nn.ConvTranspose2d(4 * n_g_feature, 2 * n_g_feature, kernel_size=4,
                       stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * n_g_feature),
    nn.ReLU(),

    nn.ConvTranspose2d(2 * n_g_feature, n_g_feature, kernel_size=4,
                       stride=2, padding=1, bias=False),
    nn.BatchNorm2d(n_g_feature),
    nn.ReLU(),

    nn.ConvTranspose2d(n_g_feature, n_channel, kernel_size=4,
                       stride=2, padding=1),
    nn.Sigmoid(),

)
print(gnet)


n_d_feature = 64
dnet = nn.Sequential(

    nn.Conv2d(n_channel, n_d_feature, kernel_size=4,
              stride=2, padding=1),
    nn.LeakyReLU(0.2),

    nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4,
              stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * n_d_feature),
    nn.LeakyReLU(0.2),

    nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4,
              stride=2, padding=1, bias=False),
    nn.BatchNorm2d(4 * n_d_feature),
    nn.LeakyReLU(0.2),

    nn.Conv2d(4 * n_d_feature, 1, kernel_size=4),

)
print(dnet)


import torch.nn.init as init


def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)



gnet.apply(weights_init)
dnet.apply(weights_init)


import torch
import torch.optim


criterion = nn.BCEWithLogitsLoss()



goptimizer = torch.optim.Adam(gnet.parameters(),
                              lr=0.0002, betas=(0.5, 0.999))
doptimizer = torch.optim.Adam(dnet.parameters(),
                              lr=0.0002, betas=(0.5, 0.999))

D_loss = []
G_loss = []

batch_size = 64
fixed_noises = torch.randn(batch_size, latent_size, 1, 1)


epoch_num = 15
for epoch in range(epoch_num):
    for batch_idx, data in enumerate(dataloader):
        if batch_idx == len(dataloader) - 1:
            continue

        real_images, _ = data


        labels = torch.ones(batch_size)
        preds = dnet(real_images)

        outputs = preds.reshape(-1)
        dloss_real = criterion(outputs, labels)
        dmean_real = outputs.sigmoid().mean()

        noises = torch.randn(batch_size, latent_size, 1, 1)
        fake_images = gnet(noises)
        labels = torch.zeros(batch_size)
        fake = fake_images.detach()
        preds = dnet(fake)
        outputs = preds.view(-1)
        dloss_fake = criterion(outputs, labels)
        dmean_fake = outputs.sigmoid().mean()


        dloss = dloss_real + dloss_fake
        dnet.zero_grad()
        dloss.backward()
        doptimizer.step()


        labels = torch.ones(batch_size)

        preds = dnet(fake_images)
        outputs = preds.view(-1)
        gloss = criterion(outputs, labels)
        gmean_fake = outputs.sigmoid().mean()

        gnet.zero_grad()
        gloss.backward()
        goptimizer.step()


        if batch_idx % 100 == 0:
            print('[{}/{}]'.format(epoch, epoch_num) +
                  '[{}/{}]'.format(batch_idx, len(dataloader)) +
                  '鉴别网络损失:{:g} 生成网络损失:{:g}'.format(dloss, gloss) +
                  '真数据判真比例:{:g} 假数据判真比例:{:g}/{:g}'.format(
                      dmean_real, dmean_fake, gmean_fake))
            G_loss.append(gloss.item())
            D_loss.append(dloss.item())
            fake = gnet(fixed_noises)
            save_image(fake,
                       './test_data1/images_epoch{:02d}_batch{:03d}.png'.format(
                           epoch, batch_idx))

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_loss,label="G")
plt.plot(D_loss,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()