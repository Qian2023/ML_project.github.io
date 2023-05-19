import os

import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.utils as vutils



img_size = 64
batch_size = 64
learn_rate = 0.0002
beta1 = 0.5
num_epoch = 10

transform_MNIST = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=0.5, std=0.5)])
transform_CIFAR10 = transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

dataset_CIFAR10 = datasets.CIFAR10(root='data', download=True,
                           transform= transform_CIFAR10)
dataset_MNIST  =  datasets.CIFAR10(root='data', download=True,
                           transform=transform_MNIST)

dataloader_CIFAR10 = torch.utils.data.DataLoader(dataset_CIFAR10, batch_size, shuffle=True)
dataloader_MNIST = torch.utils.data.DataLoader(dataset_MNIST, batch_size, shuffle=True)

# Size of latnet vector
nz = 100
# Filter size of generator
ngf = 64
# Filter size of discriminator
ndf = 64
# Output image channels
out_img_channels = 3

for batch_idx, data in enumerate(dataloader_CIFAR10):
    if batch_idx == len(dataloader_CIFAR10) - 1:
        continue
    real_images, _ = data

    print('#{} has {} images.'.format(batch_idx, batch_size))
    if batch_idx % 100 == 0:
        path = './DCGAN-CIFAR-10/CIFAR10_shuffled_batch{:03d}.png'.format(batch_idx)
        vutils.save_image(real_images, path, normalize=True)

for batch_idx, data in enumerate(dataloader_MNIST):
    if batch_idx == len(dataloader_MNIST) - 1:
        continue
    real_images, _ = data

    print('#{} has {} images.'.format(batch_idx, batch_size))
    if batch_idx % 100 == 0:
        path = './DCGAN-MNIST/MNIST_shuffled_batch{:03d}.png'.format(batch_idx)
        vutils.save_image(real_images, path, normalize=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

"Build Generator and Discreminator"
"The outputs of the hidden convolutional layers are subject to normalization operations"
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            #The generated network uses a four-layer transposed convolutional operation
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, out_img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64


        )

    def forward(self, input):
        output = self.main(input)
        return output


netG = Generator()
netG.apply(weights_init)      # By calling the apply() function, the instance of the torch.nn.Module class
                                # will recursively make itself the m of the function inside weights_init()
print(netG)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            # input is (nc) x 64 x 64
            nn.Conv2d(out_img_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()

        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


netD = Discriminator()
netD.apply(weights_init)
print(netD)

# The loss function BCEloss
criterion = nn.BCELoss()

input = torch.FloatTensor(batch_size, 3, img_size, img_size)
noise = torch.FloatTensor(batch_size, nz, 1, 1)
fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(batch_size)
real_label = 1
fake_label = 0

if torch.cuda.is_available():
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# Fixed noise for testing,
# used to see the transformation of the same latent tensor in the training process to generate images
fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizer_D = optim.Adam(netD.parameters(), learn_rate, betas=(beta1, 0.999))
optimizer_G = optim.Adam(netG.parameters(), learn_rate, betas=(beta1, 0.999))

G_loss = []
D_loss = []

for epoch in range(num_epoch):
    for batch_idx, data in enumerate(dataloader_CIFAR10, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if torch.cuda.is_available():
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)    #Variable
        labelv = Variable(label)

        # Forward pass real batch through D
        output = netD(inputv).view(-1)      #Discriminate the real data (64,1,1,1)
        # Calculate loss on all-real batch
        dloss_real = criterion(output, labelv)    # Loss of discriminator of real data
        # Calculate gradients for D in backward pass
        dloss_real.backward()
        D_x = output.data.mean()        # Calculate the percentage of true data
                                       # that the discriminator will determine to be true, for output display only

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)     #  latent noise(64,64,1,1)
        noisev = Variable(noise)                      #GPU
        # Generate fake image batch with G
        fake = netG(noisev)                    # Generate fake_data (64,3,32,32)
        labelv = Variable(label.fill_(fake_label))      # False data corresponds to a label of 0
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        dloss_fake = criterion(output, labelv)     # Discriminator loss of fake data
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        dloss_fake.backward()

        D_G_z1 = output.data.mean()             # Calculate the percentage of false data that the discriminator
                                                # will determine as true, and use it for output display only

        # Compute error of D as sum over the fake and the real batches
        errD = dloss_real + dloss_fake
        optimizer_D.step()    # Update D

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################    The generation network expects all generated data to be considered as true data

        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        #output = netD(fake)           # Putting fake data through the forensic network
        output = netD(fake).view(-1)  #  new

        # Calculate G's loss based on this output
        errG = criterion(output, labelv)          # True Data Seen Losses
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.data.mean()            # Calculate the percentage of false data
                                    # that the discriminator will determine as true, and use it for output display only
        # Update G
        optimizer_G.step()

        if batch_idx % 100 == 0:
            print('[{}/{}]'.format(epoch, num_epoch) +
                    '[{}/{}]'.format(batch_idx, len(dataloader_CIFAR10)) +
                    'Discriminator loss : {: g} Generator loss : {:g} '.format(errD, errG) +
                    'True data to true ratio: {:g} ;  Fake data to judge the true proportion: {:g}/{:g}'.format(
                        D_x, D_G_z1, D_G_z2))

            G_loss.append(errG.item())
            D_loss.append(errD.item())

            fake = netG(fixed_noise)    # False data generation from a fixed latent tensor
            vutils.save_image(fake,     # Save Fake Data
                              './DCGAN-CIFAR-10/images_epoch{:02d}_batch{:03d}.png'.format(
                                  epoch, batch_idx))



plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_loss,label="G")
plt.plot(D_loss,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
