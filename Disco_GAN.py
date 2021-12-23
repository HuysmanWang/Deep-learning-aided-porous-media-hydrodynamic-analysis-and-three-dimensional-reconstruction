import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import itertools
import sys
import datetime
import time
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image

###############需要提供的数据###############
out_folder='H:/tmp/DEEP_WATERROCK_CODE/codetest/'
dataset_name='pore2miropore'
dataset_path='H:/tmp/DEEP_WATERROCK_CODE/pore2miropore'
checkpoint_interval=10
sample_interval=10
n_epochs=20
batch_size=1
learning_rate=2e-4
channels=1
img_height=256
img_width=256
pre_trained=False
trained_epoch=0
##########################################

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.uint8(img_A)[ ::-1,:])
            img_B = Image.fromarray(np.uint8(img_B)[ ::-1,:])

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#           U-NET
##############################
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1), nn.InstanceNorm2d(out_size), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class GeneratorUNet(nn.Module):
    def __init__(self, input_shape):
        super(GeneratorUNet, self).__init__()
        channels, _, _ = input_shape
        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(128, channels, 4, padding=1), nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)

##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 3, width // 2 ** 3)

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return self.model(img)

def Disco_GAN(out_folder,
              dataset_name,
              dataset_path,
              checkpoint_interval,
              sample_interval,
              n_epochs,
              batch_size,
              lr,
              channels,
              img_height,
              img_width,
              pre_trained:bool,
              trained_epoch):
    # Create sample and checkpoint directories
    save_path=out_folder+'Image2Image_translation/'
    os.makedirs(save_path+"/disco_images/%s" % dataset_name, exist_ok=True)
    os.makedirs(save_path+"/disco_saved_models/%s" %dataset_name, exist_ok=True)
    disco_images_path=save_path+"/disco_images/%s/" % dataset_name
    disco_saved_models_path=save_path+"/disco_saved_models/%s/" %dataset_name
    # Losses
    adversarial_loss = torch.nn.MSELoss()
    cycle_loss = torch.nn.L1Loss()
    pixelwise_loss = torch.nn.L1Loss()
    cuda = torch.cuda.is_available()
    
    input_shape = (channels, img_height, img_width)
    # Initialize generator and discriminator
    G_AB = GeneratorUNet(input_shape)
    G_BA = GeneratorUNet(input_shape)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)
    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        adversarial_loss.cuda()
        cycle_loss.cuda()
        pixelwise_loss.cuda()
    
    if pre_trained==True:    
        if trained_epoch != 0:
            G_AB.load_state_dict(torch.load(disco_saved_models_path+"G_AB_%d.pth" % (trained_epoch)))
            G_BA.load_state_dict(torch.load(disco_saved_models_path+"G_BA_%d.pth" % (trained_epoch)))
            D_A.load_state_dict(torch.load(disco_saved_models_path+"D_A_%d.pth" % (trained_epoch)))
            D_B.load_state_dict(torch.load(disco_saved_models_path+"D_B_%d.pth" % (trained_epoch)))
        else:
            G_AB.apply(weights_init_normal)
            G_BA.apply(weights_init_normal)
            D_A.apply(weights_init_normal)
            D_B.apply(weights_init_normal)
    else:
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)
    
    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(0.5, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # Dataset loader
    transforms_ = [
        transforms.Resize((img_height, img_width), Image.BICUBIC),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    dataloader = DataLoader(
        ImageDataset(dataset_path, transforms_=transforms_, mode="train"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        ImageDataset(dataset_path , transforms_=transforms_, mode="val"),
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )
    
    def sample_images(batches_done,path):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(val_dataloader))
        G_AB.eval()
        G_BA.eval()
        real_A = Variable(imgs["A"].type(Tensor))
        fake_B = G_AB(real_A)
        real_B = Variable(imgs["B"].type(Tensor))
        fake_A = G_BA(real_B)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data, fake_A.data), 0)
        save_image(img_sample, path+"/%s.png" % (batches_done), nrow=8, normalize=True)
    
    # ----------
    #  Training
    # ----------
    epochs=0
    prev_time = time.time()
    for epoch in range(epochs,n_epochs):
        for i, batch in enumerate(dataloader):
            # Model inputs
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            # ------------------
            #  Train Generators
            # ------------------
            G_AB.train()
            G_BA.train()
            optimizer_G.zero_grad()
            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = adversarial_loss(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = adversarial_loss(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            # Pixelwise translation loss
            loss_pixelwise = (pixelwise_loss(fake_A, real_A) + pixelwise_loss(fake_B, real_B)) / 2
            # Cycle loss
            loss_cycle_A = cycle_loss(G_BA(fake_B), real_A)
            loss_cycle_B = cycle_loss(G_AB(fake_A), real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            # Total loss
            loss_G = loss_GAN + loss_cycle + loss_pixelwise
            loss_G.backward()
            optimizer_G.step()
            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            # Real loss
            loss_real = adversarial_loss(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = adversarial_loss(D_A(fake_A.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()
            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            # Real loss
            loss_real = adversarial_loss(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = adversarial_loss(D_B(fake_B.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
            loss_D = 0.5 * (loss_D_A + loss_D_B)
            # --------------
            #  Log Progress
            # --------------
            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, pixel: %f, cycle: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_pixelwise.item(),
                    loss_cycle.item(),
                    time_left,
                )
            )
            f=open(save_path+'disco_process.txt','a')
            f.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, pixel: %f, cycle: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_pixelwise.item(),
                    loss_cycle.item(),
                    time_left,
                )
            )
            f.close()
            # If at sample interval save image
            if batches_done % sample_interval == 0:
                sample_images(batches_done,disco_images_path)
        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), disco_saved_models_path+"G_AB_%d.pth" % (epoch))
            torch.save(G_BA.state_dict(), disco_saved_models_path+"G_BA_%d.pth" % (epoch))
            torch.save(D_A.state_dict(), disco_saved_models_path+"D_A_%d.pth" % (epoch))
            torch.save(D_B.state_dict(), disco_saved_models_path+"D_B_%d.pth" % (epoch))

'''
 13. Disco_GAN generate from promoted translation style
Disco_GAN(out_folder=out_folder,
          dataset_name=dataset_name,
          dataset_path=dataset_path,
          checkpoint_interval=checkpoint_interval,
          sample_interval=sample_interval,
          n_epochs=n_epochs,
          batch_size= batch_size,
          lr=learning_rate,
          channels=channels,
          img_height=img_height,
          img_width=img_width,
          pre_trained=pre_trained,
          trained_epoch=trained_epoch
          )
 '''
