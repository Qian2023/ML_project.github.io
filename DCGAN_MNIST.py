import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

# load the dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=0.5, std=0.5)])

train_ds = torchvision.datasets.MNIST('data/',
                                      train=True,
                                      transform=transform,
                                      download=True)
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)


# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(100, 256 * 7 * 7)
        self.bn1 = nn.BatchNorm1d(256 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(256, 128,
                                          kernel_size=(3, 3),
                                          stride=1,
                                          padding=1
                                          )  # 得到128*7*7的图像
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1  # 64*14*14
                                          )
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1  # 1*28*28
                                          )

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.bn1(x)
        x = x.view(-1, 256, 7, 7)
        x = F.relu(self.deconv1(x))
        x = self.bn2(x)
        x = F.relu(self.deconv2(x))
        x = self.bn3(x)
        x = torch.tanh(self.deconv3(x))
        return x


# 定义判别器
# input:1，28，28
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2)  #  64，13，13
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)  # 128，6，6
        self.bn = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 6 * 6, 1)  # output the probability value

    def forward(self, x):
        x = F.dropout2d(F.leaky_relu(self.conv1(x)))
        x = F.dropout2d(F.leaky_relu(self.conv2(x)))  # (batch, 128,6,6)
        x = self.bn(x)
        x = x.view(-1, 128 * 6 * 6)  # (batch, 128,6,6)--->  (batch, 128*6*6)
        x = torch.sigmoid(self.fc(x))
        return x



device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)

#loss function
loss_function = torch.nn.BCELoss()

# Define Optimizer
d_optim = torch.optim.Adam(dis.parameters(), lr=1e-5)
g_optim = torch.optim.Adam(gen.parameters(), lr=1e-4)


def generate_and_save_images(model, epoch, test_input):
    predictions = np.squeeze(model(test_input).cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] + 1) / 2, cmap='gray')
        plt.axis("off")

    plt.show()


test_input = torch.randn(16, 100, device=device)


D_loss = []
G_loss = []
# train
for epoch in range(30):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)
    # Do an iteration over the entire data set
    for step, (img, _) in enumerate(dataloader):
        img = img.to(device)
        size = img.shape[0]  # Returns the size of the first dimension of img
        random_noise = torch.randn(size, 100, device=device)

        d_optim.zero_grad()  # Zero out the gradient of the above steps
        real_output = dis(img)  # For the discriminator input real image,
        # real_output is the prediction result of the real image
        d_real_loss = loss_function(real_output,
                                    torch.ones_like(real_output, device=device)
                                    )
        d_real_loss.backward()  # Solving for gradients

        # Obtain the loss of the discriminator on the generated image
        gen_img = gen(random_noise)
        fake_output = dis(gen_img.detach())
        d_fake_loss = loss_function(fake_output,
                                    torch.zeros_like(fake_output, device=device))
        d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss
        d_optim.step()  # Optimization

        # Get the loss of the generator
        g_optim.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss_function(fake_output,
                               torch.ones_like(fake_output, device=device))
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()
    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        generate_and_save_images(gen, epoch, test_input)
    print('Epoch:', epoch)
plt.plot(D_loss, label='D_loss')
plt.plot(G_loss, label='G_loss')
plt.legend()
plt.show()