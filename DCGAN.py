#DCGAN train and generate
# 1. train
# 2. generate
import tifffile
import h5py
import torch.utils.data
from torch import Tensor
from os import listdir
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import os
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from scipy.ndimage.filters import median_filter
from skimage.filters import threshold_otsu
from collections import Counter


####################需要提供的数据FOR TRAIN####################
out_folder = 'H:/tmp/DEEP_WATERROCK_CODE/codetest/'   #数据存储路径
dataset_name='berea'
device='cuda'
manualSeed=43
imageSize=64
batchSize=32
number_generator_feature=64
number_discriminator_feature=32
number_z=512
number_train_iterations=10
number_gpu=1
####################需要提供的数据FOR GENERATE##################
seedmin=62
seedmax=64
netG='H:/tmp/DEEP_WATERROCK_CODE/codetest/DCGAN/result/netG/netG_epoch_9.pth'
generate_name='test'
image_generate_size=4
##############################################################
#判别器
class DCGAN3d_D(nn.Container):
    def __init__(self, 
                 image_size, #进入判别器中的图片大小
                 dimension_n, #nz latent space纬度
                 channel_in, #nc 进入管道数
                 D_feature_number, #ndf 判别网络中的初始feature数
                 gpu_number,
                 extra_layers_number=0):
        super(DCGAN3d_D,self).__init__()
        self.gpu_number=gpu_number
        assert image_size % 16 ==0,'image size has to be a multiple of 16'
        
        D=nn.Sequential(
            nn.Conv3d(channel_in,D_feature_number,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            )
        i=3
        next_size=image_size/2
        next_D_feature_number=D_feature_number
        
        #build next other layers
        for t in range(extra_layers_number):
            D.add_module(str(i),
                         nn.Conv3d(next_D_feature_number,
                                   next_D_feature_number,
                                   3,1,1,bias=False))
            D.add_module(str(i+1),
                         nn.BatchNorm3d(next_D_feature_number))
            D.add_module(str(i+2),
                         nn.LeakyReLU(0.2,inplace=True))
            i+=3
        while next_size>4:
            in_feat=next_D_feature_number
            out_feat=next_D_feature_number * 2
            D.add_module(str(i),
                         nn.Conv3d(in_feat,out_feat,4,2,1,bias=False))
            D.add_module(str(i+1),
                         nn.BatchNorm3d(out_feat))
            D.add_module(str(i+2),
                         nn.LeakyReLU(0.2,inplace=True))
            i+=3
            next_D_feature_number=next_D_feature_number * 2
            next_size=next_size/2
        D.add_module(str(i),
                     nn.Conv3d(next_D_feature_number,1,4,1,0,bias=False))
        D.add_module(str(i+1),
                     nn.Sigmoid())
        self.D=D
        
    def forward(self,input):
        gpu_ids=None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_number > 1:
            gpu_ids = range(self.gpu_number)
        output=nn.parallel.data_parallel(self.D,input,gpu_ids)
        return output.view(-1,1)
#生成器
class DCGAN3d_G(nn.Container):
    def __init__(self, 
                 image_size,
                 dimension_n,
                 channel_in,
                 G_feature_number, #ngf 生成网络中的初始feature数
                 gpu_number,
                 extra_layers_number=0):
        super(DCGAN3d_G,self).__init__()
        self.gpu_number=gpu_number
        assert image_size % 16 ==0, "image size has to be a multiple of 16"
        
        next_G_feature_number=G_feature_number//2
        end_image_size=4
        
        while end_image_size!=image_size:
            next_G_feature_number=next_G_feature_number * 2
            end_image_size = end_image_size * 2 
        
        G=nn.Sequential(
            nn.ConvTranspose3d(dimension_n,next_G_feature_number,4,1,0,bias=False),
            nn.BatchNorm3d(next_G_feature_number),
            nn.ReLU(True),
            )
        i=3
        next_size=4
        next_G_feature_number=next_G_feature_number
        
        while next_size<image_size//2:
            G.add_module(str(i),
                                 nn.ConvTranspose3d(next_G_feature_number,
                                                    next_G_feature_number//2,
                                                    4,2,1,bias=False))
            G.add_module(str(i+1),
                                 nn.BatchNorm3d(next_G_feature_number//2))
            G.add_module(str(i+2),
                                 nn.ReLU(True))
            i+=3
            next_G_feature_number=next_G_feature_number//2
            next_size=next_size*2
        
        #extra layers
        for t in range(extra_layers_number):
            G.add_module(str(i),
                                 nn.Conv3d(next_G_feature_number,next_G_feature_number,
                                           3,1,1,bias=False))
            G.add_module(str(i+1),
                                 nn.BatchNorm3d(next_G_feature_number))
            G.add_module(str(i+2),
                                 nn.ReLU(True))
            i+=3
        G.add_module(str(i),
                             nn.ConvTranspose3d(next_G_feature_number,channel_in,
                                                4,2,1,bias=False))
        G.add_module(str(i+1),
                             nn.Tanh())
        self.G=G
    def forward(self,input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_number> 1:
            gpu_ids = range(self.gpu_number)
        return nn.parallel.data_parallel(self.G, input, gpu_ids)
            
class DCGAN3D_G_CPU(nn.Container):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN3D_G_CPU, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d(nz, cngf, 4, 1, 0, bias=True),
            nn.BatchNorm3d(cngf),
            nn.ReLU(True),
        )

        i, csize, cndf = 3, 4, cngf
        while csize < isize//2:
            main.add_module(str(i),
                nn.ConvTranspose3d(cngf, cngf//2, 4, 2, 1, bias=True))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cngf//2))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv3d(cngf, cngf, 3, 1, 1, bias=True))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cngf))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3

        main.add_module(str(i),
                        nn.ConvTranspose3d(cngf, nc, 4, 2, 1, bias=True))
        main.add_module(str(i+1), nn.Tanh())
        self.main = main

    def forward(self, input):
        return self.main(input)

def save_hdf5(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(0.5).add(0.5).mul(255).byte().numpy()
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=ndarr, dtype="i8", compression="gzip")

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".hdf5", ".h5"])

def load_img(filepath):
    img = None
    with h5py.File(filepath, "r") as f:
        img = f['data'][()]
    img = np.expand_dims(img, axis=0)
    torch_img = Tensor(img)
    torch_img = torch_img.div(255).sub(0.5).div(0.5)
    return torch_img

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(HDF5Dataset, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = None

        return input

    def __len__(self):
        return len(self.image_filenames)

###creat training images
def train_dataset_preprocess(dataset_name):
    tiff_path=out_folder+dataset_name+'.tif'
    edge_length=64
    stride=32
    train_images_path=out_folder+'DCGAN/train_images/'
    try:
        os.makedirs(out_folder+'DCGAN/train_images')
    except OSError:
        pass
    img=tifffile.imread(tiff_path)
    N = edge_length
    M = edge_length
    O = edge_length
    I_inc = stride
    J_inc = stride
    K_inc = stride
    count = 0
    for i in range(0, img.shape[0], I_inc):
        for j in range(0, img.shape[1], J_inc):
            for k in range(0, img.shape[2], K_inc):
                subset = img[i:i+N, j:j+N, k:k+O]
                if subset.shape == (N, M, O):
                    f = h5py.File(train_images_path+"/"+str(dataset_name)+"_"+str(count)+".hdf5", "w")
                    f.create_dataset('data', data=subset, dtype="i8", compression="gzip")
                    f.close()
                    count += 1
    print('Generate images/dataset number count:',count)
    return train_images_path

def DCGAN_train(imageSize,
                batchSize,
                ngf,
                ndf,
                nz,
                niter,
                ngpu,
                manualSeed,
                out_folder,
                dataset_name,
                device):
    data_root=train_dataset_preprocess(dataset_name)
    lr=1e-5
    workers=0
    nc=1
    criterion=nn.BCELoss()
    result_path=out_folder+'DCGAN/result/'
    outf=out_folder+'DCGAN/output/'
    try:
        os.makedirs(out_folder+'DCGAN/output')
        os.makedirs(out_folder+'DCGAN/result')
    except OSError:
        pass
    np.random.seed(43)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark=True
    if torch.cuda.is_available() and device!='cuda':
        print("WARNING: You have a CUDA device, so you should probably run with device='cuda'")
    if dataset_name in ['berea']:
        dataset=HDF5Dataset(data_root,
                            input_transform=transforms.Compose([transforms.ToTensor()]))
    assert dataset
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=batchSize,shuffle=True,num_workers=int(workers))
    
    netG=DCGAN3d_G(imageSize,nz,nc,ngf,ngpu)
    netG.apply(weights_init)
    print(netG)
    netD=DCGAN3d_D(imageSize,nz,nc,ndf,ngpu)
    netD.apply(weights_init)
    print(netD)
    
    input,noise,fixed_noise,fixed_noise_TI=None,None,None,None
    input=torch.FloatTensor(batchSize,nc,imageSize,imageSize,imageSize)
    noise=torch.FloatTensor(batchSize,nz,1,1,1)
    fixed_noise=torch.FloatTensor(1,nz,7,7,7).normal_(0,1)
    fixed_noise_TI=torch.FloatTensor(1,nz,1,1,1).normal_(0,1)
    label=torch.FloatTensor(batchSize)
    real_label=0.9
    fake_label=0
    
    if device=='cuda':
        netD.cuda()
        netG.cuda()
        criterion.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        fixed_noise_TI = fixed_noise_TI.cuda()
    input = Variable(input) #变量可修改
    label = Variable(label) #变量可修改
    noise = Variable(noise) #变量可修改
    fixed_noise=Variable(fixed_noise)
    fixed_noise_TI=Variable(fixed_noise_TI)
    
    optimizerD=optim.Adam(netD.parameters(),lr=lr,betas=(0.5,0.999))
    optimizerG=optim.Adam(netG.parameters(),lr=lr,betas=(0.5,0.999))
    #main part
    gen_iterations=0
    G_loss=[]
    D_loss=[]
    iters=0
    for epoch in range(niter):
        print('This is the ',epoch,'-th')
        for i,data in enumerate(dataloader,0):
            f=open(result_path+'training_curve.scv','a')
            netD.zero_grad()
            real_cpu=data.to(device)
            batch_size=real_cpu.size(0)
            label=torch.full((batch_size,),real_label,device=device)
            output=netD(real_cpu).view(-1)
            errD_real=criterion(output,label)
            errD_real.backward()
            D_x=output.mean().item()
            
            noise=torch.randn(batch_size,nz,1,1,1,device=device)
            fake=netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            #生成器
            netG.zero_grad()
            label.fill_(1.0)
            noise2=torch.randn(batch_size,nz,1,1,1,device=device)
            fake2=netG(noise2)
            output = netD(fake2).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            gen_iterations+=1
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, niter, i, len(dataloader),
                     errD.data, errG.data, D_x, D_G_z1, D_G_z2))
            f.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, niter, i, len(dataloader),
                     errD.data, errG.data, D_x, D_G_z1, D_G_z2))
            f.write('\n')
            f.close()
        
        fake = netG(fixed_noise)
        fake_TI = netG(fixed_noise_TI)
        try:
            os.makedirs(result_path+'fake_samples')
            os.makedirs(result_path+'fake_TI')
        except OSError:
            pass
        save_hdf5(fake.data, result_path+'fake_samples/'+'fake_samples_{0}.hdf5'.format(gen_iterations))
        save_hdf5(fake_TI.data, result_path+'fake_TI/'+'fake_TI_{0}.hdf5'.format(gen_iterations))
        # do checkpointing
        try:
            os.makedirs(result_path+'netG')
            os.makedirs(result_path+'netD')
        except OSError:
            pass
        torch.save(netG.state_dict(), result_path+'netG/'+'netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), result_path+'netD/'+'netD_epoch_%d.pth' % (epoch))
        #record loss
        G_loss.append(errG.item())
        D_loss.append(errD.item())
        iters+=1
    f=open(result_path+'Loss_log.txt','a')
    f.write('G_loss:')
    f.write('\n')
    for k in range(len(G_loss)):
        f.write(str(G_loss[k]))
        f.write('\n')
    f.write('D_loss:')
    f.write('\n')
    for k in range(len(D_loss)):
        f.write(str(D_loss[k]))
        f.write('\n')
    f.close()
                   
def DCGAN_generator(seedmin,
                    seedmax,
                    ngf,
                    ndf,
                    nz,
                    ngpu,
                    imageSize,
                    imsize,
                    out_folder,
                    name,
                    device,
                    netG,
                    ):

    if name is None:
        name = 'samples'
    try:
        os.makedirs(out_folder+'DCGAN/output/'+name)
    except OSError:
        pass
    outf=out_folder+'DCGAN/output/'
       
    for seed in range(seedmin, seedmax, 1):
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.benchmark = True
        ngpu = int(ngpu)
        nz = int(nz)
        ngf = int(ngf)
        ndf = int(ndf)
        nc = 1
        
        net = DCGAN3d_G(imageSize, nz, nc, ngf, ngpu)
        net.apply(weights_init)
        net.load_state_dict(torch.load(netG))
        print(net)
    
        fixed_noise = torch.FloatTensor(1, nz, imsize, imsize, imsize).normal_(0, 1)
        if device=='cuda':
            net.cuda()
            fixed_noise = fixed_noise.cuda()
        fixed_noise = Variable(fixed_noise)
        fake = net(fixed_noise)
        save_hdf5(fake.data, '{0}/{1}_{2}.hdf5'.format(outf+name, name, seed))

def result_analysis(out_folder,generate_name):
    path=out_folder+'DCGAN/output/'+generate_name+'/'
    tiff_name=generate_name+'_tiff'
    datalist=os.listdir(path)
    try:
        os.makedirs(out_folder+'DCGAN/output/'+tiff_name)
    except OSError:
        pass
    for img in datalist:
        f=h5py.File(path+img,'r')
        array=f['data'][()]
        tiff=array[0,0,:,:,:].astype(np.float32)
        tifffile.imsave(out_folder+'DCGAN/output/{0}/{1}.tiff'.format(tiff_name,img[:-5]),tiff)
        
    path2=out_folder+'DCGAN/output/'+tiff_name
    tifflist=os.listdir(path2)
    for img in tifflist:
        f=open(out_folder+'DCGAN/output/'+generate_name+'_log.txt','a')
        im_in=tifffile.imread(path2+'/'+img)
        im_in=median_filter(im_in,size=(3,3,3))
        im_in=im_in[40:240,40:240,40:240]
        im_in=im_in/255.
        threshold_global_otsu=threshold_otsu(im_in)
        segmented_image=(im_in>=threshold_global_otsu).astype(np.int32)
        porc=Counter(segmented_image.flatten())
        porosity=porc[0]/(porc[0]+porc[1])
        print(img[:-5],' porosity: ',porosity)
        f.write(str(img[:-5])+'  porosity: '+str(porosity))
        f.write('\n')
        f.close()
'''
 9. DCGAN train

DCGAN_train(imageSize=imageSize,batchSize=batchSize,
                ngf=number_generator_feature,
                ndf=number_discriminator_feature,
                nz=number_z,
                niter=number_train_iterations,
                ngpu=number_gpu,
                manualSeed=manualSeed,
                out_folder=out_folder,
                dataset_name=dataset_name,
                device=device)

'''
'''
 10. DCGAN generate

DCGAN_generator(seedmin=seedmin,
                seedmax=seedmax,
                ngf=number_generator_feature,
                ndf=number_discriminator_feature,
                nz=number_z,
                ngpu=number_gpu,
                imageSize=imageSize,
                imsize=image_generate_size,
                out_folder=out_folder,
                name=generate_name,
                device=device,
                netG=netG,
                    )
'''
'''
  11. DCGAN batch processing samples statistic
  
result_analysis(out_folder,generate_name)
'''
