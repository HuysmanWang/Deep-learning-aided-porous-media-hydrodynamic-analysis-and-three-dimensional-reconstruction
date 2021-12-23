import random
import time
import datetime
import sys
import cv2
import itertools
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
import ot
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

###############需要提供的数据###############
#Cycle_GAN
out_folder='H:/tmp/DEEP_WATERROCK_CODE/codetest/'
dataset_name='pore2miropore_cycle'
dataset_path='H:/tmp/DEEP_WATERROCK_CODE/pore2miropore_cycle'
n_epochs=10
decay_epoch=8
batch_size=1
learning_rate=2e-5
Resnet_blocks=9
img_height=256
img_width=256
channels=3
sample_interval=5
checkpoint_interval=1
pre_trained=False
trained_epoch=0

#SWD
cross_number=[1,10,100]
berea_calc_WD_SWD_datasetpath='H:/tmp/DEEP_WATERROCK_CODE/pore2miropore_cycle/train/B/'
test_loader='H:/tmp/DEEP_WATERROCK_CODE/test_loader'

#test
G_AB_path='H:/清华大学/论文/论文《DEEP FLOW》王明阳/深度分割重建计算/study_gan_5/229_cross_result/model/dataset_100/G_AB_29.pth'
G_BA_path='H:/清华大学/论文/论文《DEEP FLOW》王明阳/深度分割重建计算/study_gan_5/229_cross_result/model/dataset_100/G_BA_29.pth'
test_result_save_path=out_folder+'/Image2Image_translation'
##########################################

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
    
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    

def Cycle_GAN(out_folder,
              dataset_name,
              dataset_path,
              checkpoint_interval,
              sample_interval,
              n_epochs,
              batch_size,
              lr,
              decay_epoch,
              n_residual_blocks,
              channels,
              img_height,
              img_width,
              pre_trained:bool,
              trained_epoch=0):
    lambda_cyc=10.0
    lambda_id=5.0
    b1=0.5
    b2=0.999
    n_cpu=0
    # Create sample and checkpoint directories
    save_path=out_folder+'Image2Image_translation/'
    os.makedirs(save_path+'cycle_images/%s' % dataset_name, exist_ok=True)
    os.makedirs(save_path+'cycle_saved_models/%s' % dataset_name, exist_ok=True)
    cycle_images_path=save_path+'cycle_images/%s' % dataset_name
    cycle_saved_models_path=save_path+'cycle_saved_models/%s' % dataset_name
    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    cuda = torch.cuda.is_available()
    input_shape = (channels, img_height, img_width)
    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shape, n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)
    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()
    if pre_trained==True:    
        if trained_epoch != 0:
            G_AB.load_state_dict(torch.load(cycle_saved_models_path+"G_AB_%d.pth" % (trained_epoch)))
            G_BA.load_state_dict(torch.load(cycle_saved_models_path+"G_BA_%d.pth" % (trained_epoch)))
            D_A.load_state_dict(torch.load(cycle_saved_models_path+"D_A_%d.pth" % (trained_epoch)))
            D_B.load_state_dict(torch.load(cycle_saved_models_path+"D_B_%d.pth" % (trained_epoch)))
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
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))
    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(n_epochs, trained_epoch, decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(n_epochs, trained_epoch, decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(n_epochs, trained_epoch, decay_epoch).step
    )
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    # Image transformations
    transforms_ = [
        transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
        transforms.RandomCrop((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    # Training data loader
    dataloader = DataLoader(
        ImageDataset(dataset_path , transforms_=transforms_, unaligned=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    # Test data loader
    val_dataloader = DataLoader(
        ImageDataset(dataset_path, transforms_=transforms_, unaligned=True, mode="val"),
        batch_size=5,
        shuffle=True,
        num_workers=0,
    )
    
    def sample_images(batches_done):
        """Saves a generated sample from the test set"""
        imgs = next(iter(val_dataloader))
        G_AB.eval()
        G_BA.eval()
        real_A = Variable(imgs["A"].type(Tensor))
        fake_B = G_AB(real_A)
        real_B = Variable(imgs["B"].type(Tensor))
        fake_A = G_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        save_image(image_grid, cycle_images_path+"/%s.png" % (batches_done), normalize=False)
    # ----------
    #  Training
    # ----------
    prev_time = time.time()
    for epoch in range(trained_epoch, n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
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
            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2
            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            # Total loss
            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
            loss_G.backward()
            optimizer_G.step()
            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()
            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
            loss_D = (loss_D_A + loss_D_B) / 2
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
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )
            f=open(save_path+'cycle_process.txt','a')
            f.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )
            f.close()
            # If at sample interval save image
            if batches_done % sample_interval == 0:
                sample_images(batches_done)
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), cycle_saved_models_path+"G_AB_%d.pth" % (epoch))
            torch.save(G_BA.state_dict(), cycle_saved_models_path+"G_BA_%d.pth" % (epoch))
            torch.save(D_A.state_dict(), cycle_saved_models_path+"D_A_%d.pth" % (epoch))
            torch.save(D_B.state_dict(), cycle_saved_models_path+"D_B_%d.pth" % (epoch))
        
'''
用于cross SWD cycle GAN
'''
def get_random_projections(n_projections, d, seed=None):
    if not isinstance(seed, np.random.RandomState):
        random_state = np.random.RandomState(seed)
    else:
        random_state = seed
    projections = random_state.normal(0., 1., [n_projections, d])
    norm = np.linalg.norm(projections, ord=2, axis=1, keepdims=True)
    projections = projections / norm
    return projections

def sliced_wasserstein_distance(X_s, X_t, a=None, b=None, n_projections=50, seed=None, log=False):
    from ot.lp import emd2_1d
    X_s = np.asanyarray(X_s)
    X_t = np.asanyarray(X_t)
    n = X_s.shape[0]
    m = X_t.shape[0]
    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(X_s.shape[1],X_t.shape[1]))
    if a is None:
        a = np.full(n, 1 / n)
    if b is None:
        b = np.full(m, 1 / m)
    d = X_s.shape[1]
    projections = get_random_projections(n_projections, d, seed)
    X_s_projections = np.dot(projections, X_s.T)
    X_t_projections = np.dot(projections, X_t.T)
    if log:
        projected_emd = np.empty(n_projections)
    else:
        projected_emd = None
    res = 0.
    for i, (X_s_proj, X_t_proj) in enumerate(zip(X_s_projections, X_t_projections)):
        emd = emd2_1d(X_s_proj, X_t_proj, a, b, log=False, dense=False)
        if projected_emd is not None:
            projected_emd[i] = emd
        res += emd
    res = (res / n_projections) ** 0.5
    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res

def get_fid(img_A,img_B):
    import numpy as np
    from numpy import cov
    from scipy.linalg import sqrtm
    mu1,sigma1=img_A.mean(axis=0),cov(img_A,rowvar=False)
    mu2,sigma2=img_B.mean(axis=0),cov(img_B,rowvar=False)
    ssdiff=np.sum((mu1-mu2)**2.0)
    covmean=sqrtm(sigma1.dot(sigma2))
    fid=ssdiff+np.trace(sigma1+sigma2-2.0*covmean)
    return fid

def images_style_index(img_A:str,img_B:str):
    imgA=np.asarray(Image.open(img_A))
    imgB=np.asarray(Image.open(img_B))
    fid_1=get_fid(imgA,imgB)
    wd_1=wasserstein_distance(imgA.flatten(),imgB.flatten())
    swd_1=sliced_wasserstein_distance(imgA,imgB)
    print('Fid:',fid_1)
    print('Wasserstrin distance:',wd_1)
    print('Sliced Wasserstein distance:',swd_1)
    result={'Fid:':fid_1,
            'Wasserstrin distance:':wd_1,
            'Sliced Wasserstein distance:':swd_1}
    return result

def WD_SWD_calc(path):
    WD=[]
    SWD=[]
    samples=os.listdir(path)
    number=[]
    for num in samples:
        num=num[:-4]
        number.append(num)
    number=sorted(np.array(number).astype('int32')) #number is the number of images low to high
    for i in range(len(number)-1):
        img_forward=np.asarray(Image.open(path+str(i)+'.png'))
        img_backward=np.asarray(Image.open(path+str(i+1)+'.png'))
        wd=wasserstein_distance(img_forward.flatten(),img_backward.flatten())
        WD.append(wd)
        swd=sliced_wasserstein_distance(img_forward,img_backward)
        SWD.append(swd)
    return WD,SWD

def WD_SWD_distribution_plot(WD,SWD,save_path,dataset_name):
    n=len(SWD)
    X=np.arange(0,n,1)
    plt.figure(figsize=(30,6))
    plt.plot(X,WD,'b')
    plt.fill_between(X, y1=0, y2=WD, facecolor='red', alpha=0.6)
    plt.title('Wassertein distance distribution',fontsize=35)
    plt.grid()
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.savefig(save_path+dataset_name+'_WD_distribution.png',dpi=100,bbox_inches='tight')
    
    plt.figure(figsize=(30,6))
    plt.plot(X,SWD,'b')
    plt.fill_between(X, y1=8.3, y2=SWD, facecolor='red', alpha=0.6)
    plt.title('Sliced wasserstein distance distribution',fontsize=35)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid()
    plt.savefig(save_path+dataset_name+'_SWD_distribution.png',dpi=100,bbox_inches='tight')
    
def cross_cycle_dataset(original_path,out_folder,cross_number:list):
    save_path=out_folder+'cross_datasets/'
    
    
    original_train_A=original_path+'/train/A/'
    original_train_B=original_path+'/train/B/'
    original_val_A=original_path+'/val/A/'
    original_val_B=original_path+'/val/B/'
    #取train顺序
    samples=os.listdir(original_train_A)
    original_img_sequence=[]
    for num in samples:
        num=num[:-4]
        original_img_sequence.append(num)
    original_img_sequence=sorted(np.array(original_img_sequence).astype('int32')) #original name sequence
    #取val顺序
    samples2=os.listdir(original_val_A)
    original_img2_sequence=[]
    for num in samples2:
        num=num[:-4]
        original_img2_sequence.append(num)
    original_img2_sequence=sorted(np.array(original_img2_sequence).astype('int32'))
    
    n=min(len(original_img_sequence),len(original_img2_sequence))
    if cross_number==[]:
        for i in range(n):
            #train/A/
            os.makedirs(save_path+'dataset_'+str(i)+'/train/A/',exist_ok=True) #mode/A/
            new_sequence_trainA=list(np.roll(np.array(original_img_sequence),i+1)) #adjust name sequence
            new_path_trainA=save_path+'dataset_'+str(i)+'/train/A/'
            for j in range(len(original_img_sequence)):
                image=cv2.imread(original_train_A+str(original_img_sequence[j])+'.png')
                cv2.imwrite(new_path_trainA+str(new_sequence_trainA[j])+'.png',image)
            #train/B/
            os.makedirs(save_path+'dataset_'+str(i)+'/train/B/',exist_ok=True)
            new_path_trainB=save_path+'dataset_'+str(i)+'/train/B/'
            for j in range(len(original_img_sequence)):
                image=cv2.imread(original_train_B+str(original_img_sequence[j])+'.png')
                cv2.imwrite(new_path_trainB+str(original_img_sequence[j])+'.png',image)
                 
            #val/A/
            os.makedirs(save_path+'dataset_'+str(i)+'/val/A/',exist_ok=True) #mode/A/
            new_sequence_valA=list(np.roll(np.array(original_img2_sequence),i+1)) #adjust name sequence
            new_path_valA=save_path+'dataset_'+str(i)+'/val/A/'
            for j in range(len(original_img2_sequence)):
                image=cv2.imread(original_val_A+str(original_img2_sequence[j])+'.png')
                cv2.imwrite(new_path_valA+str(new_sequence_valA[j])+'.png',image)
            #val/B/
            os.makedirs(save_path+'dataset_'+str(i)+'/val/B/',exist_ok=True)
            new_path_valB=save_path+'dataset_'+str(i)+'/val/B/'
            for j in range(len(original_img2_sequence)):
                image=cv2.imread(original_val_B+str(original_img2_sequence[j])+'.png')
                cv2.imwrite(new_path_valB+str(original_img2_sequence[j])+'.png',image)       
                
    else:
        for number in cross_number:
            os.makedirs(save_path+'dataset_'+str(number)+'/train/A/',exist_ok=True)
            new_sequence_trainA=list(np.roll(np.array(original_img_sequence),number))
            new_path_trainA=save_path+'dataset_'+str(number)+'/train/A/'
            os.makedirs(save_path+'dataset_'+str(number)+'/val/A/',exist_ok=True) #mode/A/
            new_sequence_valA=list(np.roll(np.array(original_img2_sequence),number)) #adjust name sequence
            new_path_valA=save_path+'dataset_'+str(number)+'/val/A/'
            for j in range(len(original_img_sequence)):
                image=cv2.imread(original_train_A+str(original_img_sequence[j])+'.png')
                cv2.imwrite(new_path_trainA+str(new_sequence_trainA[j])+'.png',image)
            for j in range(len(original_img2_sequence)):
                image=cv2.imread(original_val_A+str(original_img2_sequence[j])+'.png')
                cv2.imwrite(new_path_valA+str(new_sequence_valA[j])+'.png',image)
                
            os.makedirs(save_path+'dataset_'+str(number)+'/train/B/',exist_ok=True) 
            new_path_trainB=save_path+'dataset_'+str(number)+'/train/B/'
            os.makedirs(save_path+'dataset_'+str(number)+'/val/B/',exist_ok=True) 
            new_path_valB=save_path+'dataset_'+str(number)+'/val/B/'
            for j in range(len(original_img_sequence)):
                image=cv2.imread(original_train_B+str(original_img_sequence[j])+'.png')
                cv2.imwrite(new_path_trainB+str(original_img_sequence[j])+'.png',image)
            for j in range(len(original_img2_sequence)):
                image=cv2.imread(original_train_B+str(original_img2_sequence[j])+'.png')
                cv2.imwrite(new_path_valB+str(original_img2_sequence[j])+'.png',image)
                
    return print('All dataset generate done')

 
def testloader_result(test_loader,n_residual_blocks,G_AB_path,G_BA_path,test_result_save_path,channels,img_height,img_width):
    input_shape = (channels, img_height, img_width)
    G_AB = GeneratorResNet(input_shape, n_residual_blocks).cuda()
    G_BA = GeneratorResNet(input_shape, n_residual_blocks).cuda()
    G_AB.load_state_dict(torch.load(G_AB_path))
    G_BA.load_state_dict(torch.load(G_BA_path))
    
    
    cuda = torch.cuda.is_available()
    transforms_ = [
        transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
        transforms.RandomCrop((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    test_loader=DataLoader(
        ImageDataset(test_loader, transforms_=transforms_, unaligned=True, mode="test"),
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    for i,batch in enumerate(test_loader):
        real_A=Variable(batch['A'].type(Tensor))
        real_B=Variable(batch['B'].type(Tensor))
        
        fake_B = G_AB(real_A).cpu().detach().numpy()
        fake_A=G_BA(real_B).cpu().detach().numpy()
        plt.figure(figsize=(16,4))
        plt.subplot(1,4,1)
        plt.title('G_AB')
        plt.imshow(fake_B[0,0,:,:],cmap=plt.cm.gray)
        plt.subplot(1,4,2)
        plt.title('G_BA')
        plt.imshow(fake_A[0,0,:,:],cmap=plt.cm.gray)
        plt.subplot(1,4,3)
        plt.title('GT_A')
        plt.imshow(batch['A'][0,0,:,:].type(Tensor).cpu().detach().numpy(),cmap=plt.cm.gray)
        plt.subplot(1,4,4)
        plt.title('GT_B')
        plt.imshow(batch['B'][0,0,:,:].type(Tensor).cpu().detach().numpy(),cmap=plt.cm.gray)
        plt.savefig(test_result_save_path+'/generate_result.png',dpi=100,bbox_inches='tight')

def Cross_Cycle_GAN(out_folder,
                    cross_number,
              checkpoint_interval,
              sample_interval,
              n_epochs,
              batch_size,
              lr,
              decay_epoch,
              n_residual_blocks,
              channels,
              img_height,
              img_width,
              pre_trained:bool,
              trained_epoch=0):
    lambda_cyc=10.0
    lambda_id=5.0
    b1=0.5
    b2=0.999
    n_cpu=0
    # Create sample and checkpoint directories
    save_path=out_folder+'3D_Reconstruction/'
    
    def sample_images(batches_done):
        """Saves a generated sample from the test set"""
        imgs = next(iter(val_dataloader))
        G_AB.eval()
        G_BA.eval()
        real_A = Variable(imgs["A"].type(Tensor))
        fake_B = G_AB(real_A)
        real_B = Variable(imgs["B"].type(Tensor))
        fake_A = G_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        save_image(image_grid, cross_images_path+"/%s.png" % (batches_done), normalize=False)
    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    cuda = torch.cuda.is_available()
    input_shape = (channels, img_height, img_width)
    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shape, n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)
    
    for number in cross_number:
        dataset_path=out_folder+'cross_datasets/dataset_%s'%number
        os.makedirs(save_path+'cross_images/dataset_%s' % number, exist_ok=True)
        os.makedirs(save_path+'cross_models/dataset_%s' % number, exist_ok=True)
        cross_images_path=save_path+'cross_images/dataset_%s' % number
        cross_saved_models_path=save_path+'cross_models/dataset_%s' % number
 
        if cuda:
            G_AB = G_AB.cuda()
            G_BA = G_BA.cuda()
            D_A = D_A.cuda()
            D_B = D_B.cuda()
            criterion_GAN.cuda()
            criterion_cycle.cuda()
            criterion_identity.cuda()
        if pre_trained==True:    
            if trained_epoch != 0:
                G_AB.load_state_dict(torch.load(cross_saved_models_path+"G_AB_%d.pth" % (trained_epoch)))
                G_BA.load_state_dict(torch.load(cross_saved_models_path+"G_BA_%d.pth" % (trained_epoch)))
                D_A.load_state_dict(torch.load(cross_saved_models_path+"D_A_%d.pth" % (trained_epoch)))
                D_B.load_state_dict(torch.load(cross_saved_models_path+"D_B_%d.pth" % (trained_epoch)))
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
            itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
        )
        optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))
        # Learning rate update schedulers
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            optimizer_G, lr_lambda=LambdaLR(n_epochs, trained_epoch, decay_epoch).step
        )
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_A, lr_lambda=LambdaLR(n_epochs, trained_epoch, decay_epoch).step
        )
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_B, lr_lambda=LambdaLR(n_epochs, trained_epoch, decay_epoch).step
        )
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        # Buffers of previously generated samples
        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()
        # Image transformations
        transforms_ = [
            transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
            transforms.RandomCrop((img_height, img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        # Training data loader
        dataloader = DataLoader(
            ImageDataset(dataset_path , transforms_=transforms_, unaligned=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        # Test data loader
        val_dataloader = DataLoader(
            ImageDataset(dataset_path, transforms_=transforms_, unaligned=True, mode="val"),
            batch_size=5,
            shuffle=True,
            num_workers=0,
        )
        # ----------
        #  Training
        # ----------
        prev_time = time.time()
        for epoch in range(trained_epoch, n_epochs):
            for i, batch in enumerate(dataloader):
                # Set model input
                f=open(save_path+'dataset_%s_cross_log.txt'%number,'a')
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
                # Identity loss
                loss_id_A = criterion_identity(G_BA(real_A), real_A)
                loss_id_B = criterion_identity(G_AB(real_B), real_B)
                loss_identity = (loss_id_A + loss_id_B) / 2
                
                calc1=G_BA(real_A).cpu().detach().numpy()[0,0,:,:]
                calc2=real_A.cpu().detach().numpy()[0,0,:,:]
                wd_1=wasserstein_distance(calc1.flatten(),calc2.flatten()) 
                calc3=G_AB(real_B).cpu().detach().numpy()[0,0,:,:]
                calc4=real_B.cpu().detach().numpy()[0,0,:,:]
                wd_2=wasserstein_distance(calc3.flatten(),calc4.flatten())
                wd=(wd_1+wd_2)/2
                
                swd_1=sliced_wasserstein_distance(calc1,calc2)
                swd_2=sliced_wasserstein_distance(calc3,calc4)
                swd=(swd_1+swd_2)/2
                
                # GAN loss
                fake_B = G_AB(real_A)
                loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
                fake_A = G_BA(real_B)
                loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
                loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
                # Cycle loss
                recov_A = G_BA(fake_B)
                loss_cycle_A = criterion_cycle(recov_A, real_A)
                recov_B = G_AB(fake_A)
                loss_cycle_B = criterion_cycle(recov_B, real_B)
                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
                # Total loss
                loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
                loss_G.backward()
                optimizer_G.step()
                # -----------------------
                #  Train Discriminator A
                # -----------------------
                optimizer_D_A.zero_grad()
                # Real loss
                loss_real = criterion_GAN(D_A(real_A), valid)
                # Fake loss (on batch of previously generated samples)
                fake_A_ = fake_A_buffer.push_and_pop(fake_A)
                loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
                # Total loss
                loss_D_A = (loss_real + loss_fake) / 2
                loss_D_A.backward()
                optimizer_D_A.step()
                # -----------------------
                #  Train Discriminator B
                # -----------------------
                optimizer_D_B.zero_grad()
                # Real loss
                loss_real = criterion_GAN(D_B(real_B), valid)
                # Fake loss (on batch of previously generated samples)
                fake_B_ = fake_B_buffer.push_and_pop(fake_B)
                loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
                # Total loss
                loss_D_B = (loss_real + loss_fake) / 2
                loss_D_B.backward()
                optimizer_D_B.step()
                loss_D = (loss_D_A + loss_D_B) / 2
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
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s [WD: %f, SWD: %f]"
                    % (
                        epoch,
                        n_epochs,
                        i,
                        len(dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_GAN.item(),
                        loss_cycle.item(),
                        loss_identity.item(),
                        time_left,
                        wd,
                        swd,
                    )
                )
                
                f.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s [WD: %f, SWD: %f]"
                    % (
                        epoch,
                        n_epochs,
                        i,
                        len(dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_GAN.item(),
                        loss_cycle.item(),
                        loss_identity.item(),
                        time_left,
                        wd,
                        swd,
                    )
                )
                f.write('\n')
                f.close()
                # If at sample interval save image
                if batches_done % sample_interval == 0:
                    imgs = next(iter(val_dataloader))
                    G_AB.eval()
                    G_BA.eval()
                    real_A = Variable(imgs["A"].type(Tensor))
                    fake_B = G_AB(real_A)
                    real_B = Variable(imgs["B"].type(Tensor))
                    fake_A = G_BA(real_B)
                    real_A = make_grid(real_A, nrow=5, normalize=True)
                    real_B = make_grid(real_B, nrow=5, normalize=True)
                    fake_A = make_grid(fake_A, nrow=5, normalize=True)
                    fake_B = make_grid(fake_B, nrow=5, normalize=True)
                    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
                    save_image(image_grid, cross_images_path+'/%s.png' % batches_done, normalize=False)
            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()
            if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(G_AB.state_dict(), cross_saved_models_path+"G_AB_%d.pth" % (epoch))
                torch.save(G_BA.state_dict(), cross_saved_models_path+"G_BA_%d.pth" % (epoch))
                torch.save(D_A.state_dict(), cross_saved_models_path+"D_A_%d.pth" % (epoch))
                torch.save(D_B.state_dict(), cross_saved_models_path+"D_B_%d.pth" % (epoch))

def SWD_cross_cycle(out_folder,
                    n_epochs,
                    channels,
                    img_height,
                    img_width,
                    n_residual_blocks,
                    cross_number,
                    berea_calc_WD_SWD_datasetpath,
                    test_loader,
                    ):
    n_residual_blocks=Resnet_blocks
    n_epochs=n_epochs
    channels=channels
    img_height=img_height
    img_width=img_width
    out_folder=out_folder
    models_path=out_folder+'3D_Reconstruction/cross_models/'
    save_path=out_folder+'3D_Reconstruction/cross_generate/'
    os.makedirs(save_path, exist_ok=True)
    
    models=cross_number
    
    WD_berea,SWD_berea=WD_SWD_calc(berea_calc_WD_SWD_datasetpath)
    layer=len(SWD_berea) #399
    SWD_new=[]
    input_shape = (channels, img_height, img_width)
    G_AB = GeneratorResNet(input_shape, n_residual_blocks).cuda()
    G_BA = GeneratorResNet(input_shape, n_residual_blocks).cuda()
    cuda=torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    
    for l in range(layer):
        swd=0
        if l==0:
            for i in models:
                if i==1:
                    break
                else:
                    G_BA.load_state_dict(torch.load(models_path+'/dataset_%s/G_AB_%s.pth'%(i,n_epochs-1)))
                    for j,batch in enumerate(test_loader):
                        real_A=Variable(batch['A'].type(Tensor))
                        real_B=Variable(batch['B'].type(Tensor))
                        fake_A=G_BA(real_B)
                        real_AA=real_A.cpu().detach().numpy()#用来计算swd
                        fake_AA=fake_A.cpu().detach().numpy()#用来计算swd
                        swd=sliced_wasserstein_distance(fake_AA[0,0,:,:],real_AA[0,0,:,:])
                        SWD_new.append(swd)
                        save_image(fake_A,save_path+str(l+1)+'_layer.png',normalize=False)
                    break
        else:
            for i in models:
                ts=0
                G_BA.load_state_dict(torch.load(torch.load(models_path+'/dataset_%s/G_AB_%s.pth'%(i,n_epochs-1))))
                while -2>np.abs(swd-SWD_berea[l])>2 and ts<100:
                    for j,batch in enumerate(test_loader):
                        real_A=Variable(batch['A'].type(Tensor))
                        real_B=Variable(batch['B'].type(Tensor))
                        fake_A=G_BA(real_B)
                        real_AA=real_A.cpu().detach().numpy()#用来计算swd
                        fake_AA=fake_A.cpu().detach().numpy()#用来计算swd
                        save_image(fake_A,save_path+str(l+1)+'_layer.png',normalize=False)
                        last_layer=np.asarray(Image.open(save_path+str(l)+'_layer.png'))
                        this_layer=np.asarray(Image.open(save_path+str(l+1)+'_layer.png'))
                        swd=sliced_wasserstein_distance(this_layer[0,:,:],last_layer[0,:,:])
                    ts+=1
                SWD_new.append(swd)
                break
    return SWD_new

'''
 14. Cycle_GAN generate from promoted translation style
Cycle_GAN(out_folder=out_folder,
          dataset_name=dataset_name,
          dataset_path=dataset_path,
          checkpoint_interval=checkpoint_interval,
          sample_interval=sample_interval,
          n_epochs=n_epochs,
          batch_size=batch_size,
          lr=learning_rate,
          decay_epoch=decay_epoch,
          n_residual_blocks=Resnet_blocks,
          channels=channels,
          img_height=img_height,
          img_width=img_width,
          pre_trained=pre_trained,
          trained_epoch=trained_epoch
          )
 '''
'''
 15. SWD WS and FID distribution calc
WD,SWD=WD_SWD_calc(berea_calc_WD_SWD_datasetpath)
WD_SWD_distribution_plot(WD,SWD,save_path=out_folder,dataset_name='berea')
'''
'''
 16. corss domain datasets create
cross_cycle_dataset(original_path=dataset_path,
                    out_folder=out_folder,
                    cross_number=cross_number)
 '''
'''
 17. cross datasets train models  #一个模型一个模型地训练，要清除变量
 cross train:
Cross_Cycle_GAN(out_folder=out_folder,
                cross_number=cross_number,
                checkpoint_interval=checkpoint_interval,
                sample_interval=sample_interval,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=learning_rate,
                decay_epoch=decay_epoch,
                n_residual_blocks=Resnet_blocks,
                channels=channels,
                img_height=img_height,
                img_width=img_width,
                pre_trained=pre_trained,
                trained_epoch=trained_epoch)
 testloader visualization:
testloader_result(test_loader=test_loader,
                  n_resudual_blocks=Resnet_blocks,
                  G_AB_path=G_AB_path,
                  G_BA_path=G_BA_path,
                  test_result_save_path=test_result_save_path,
                  channels=channels,
                  img_height=img_height,
                  img_width=img_width)
 '''
'''
 18. SWD-guided Cycle-GAN 3D reconstruction
Generate_SWD=SWD_cross_cycle(out_folder=out_folder, 
                             n_epochs=n_epochs,
                             channels=channels,
                             img_height=img_height,
                             img_width=img_width,
                             n_residual_blocks=Resnet_blocks,
                             cross_number=cross_number,
                             berea_calc_WD_SWD_datasetpath=berea_calc_WD_SWD_datasetpath,
                             test_loader=test_loader)
 
 '''
