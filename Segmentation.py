#Segmentation
# 1. train and test
# 2. test and viz using trained model parameters

import os
import torch 
import json
import labelme
import numpy as np
import cv2
import torch.nn as nn
from PIL import Image
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
from Image_preprocess import *

#from Augmentation import *

####################需要提供的数据####################
out_folder = 'H:/tmp/DEEP_WATERROCK_CODE/codetest/'   #数据存储路径
label_number=5 #想要标记图片的数量
json_path='H:/tmp/json/'  #labelme之后翻译得到的地址,路径不要出现中文
dataset_choose='obstacle' #check in dataset,choose what you want to extract
#####################################################

###############训练过程中需要定义的参数##############
model_name='Unet'
Encoders=['resnet18','vgg16']
Activation='sigmoid'    #sigmoid,relu,tanh
Encoder_weights ='imagenet'
Epoch=10
train_batch_size=3
#####################################################

class Dataset(BaseDataset):
    CLASSES = ['obstacle','soil','media']
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask
    def __len__(self):
        return len(self.ids)
    
class Dataset2(BaseDataset):
    CLASSES = ['obstacle','soil','media']
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask
    def __len__(self):
        return len(self.ids)
    
def visualize(**images):
    n = len(images)
    plt.figure(figsize=(8,4))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image,cmap=plt.cm.gray)
    plt.show()
    
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=800, min_width=800, always_apply=True, border_mode=0),
        albu.RandomCrop(height=800, width=800, always_apply=True),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(800,800)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2,0,1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def Model_create(model_name:str,Encoder_name:str,Encoder_weights:str,CLASSES,Activation:str):
    Models=['DeeplabV3','Deeplabv3+','FPN','Linknet','Unet','Unet++']
    if model_name=='DeeplabV3':
          model=smp.DeepLabV3(
              encoder_name=Encoder_name,
              encoder_weights=Encoder_weights,
              classes=len(CLASSES),
              activation=Activation,)
    elif model_name=='Deeplabv3+':
        model=smp.DeepLabV3Plus(
              encoder_name=Encoder_name,
              encoder_weights=Encoder_weights,
              classes=len(CLASSES),
              activation=Activation,)
    elif model_name=='FPN':
        model=smp.FPN(
              encoder_name=Encoder_name,
              encoder_weights=Encoder_weights,
              classes=len(CLASSES),
              activation=Activation,)
    elif model_name=='Linknet':
        model=smp.Linknet(
              encoder_name=Encoder_name,
              encoder_weights=Encoder_weights,
              classes=len(CLASSES),
              activation=Activation,)
    elif model_name=='Unet':
        model=smp.Unet(
              encoder_name=Encoder_name,
              encoder_weights=Encoder_weights,
              classes=len(CLASSES),
              activation=Activation,)
    elif model_name=='Unet++':
        model=smp.UnetPlusPlus(
              encoder_name=Encoder_name,
              encoder_weights=Encoder_weights,
              classes=len(CLASSES),
              activation=Activation,)
    else:
        print('Please select a model from the following list\n',Models)
    return model

def cuda_is_available():
    if torch.cuda.is_available():
        gpu_num=torch.cuda.device_count()
        if gpu_num==1:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            device='cuda'
            device_ids=[0]
        elif gpu_num==2:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
            device='cuda'
            device_ids=[0,1]
        elif gpu_num==3:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
            device='cuda'
            device_ids=[0,1,2]
    else:
        device='cpu'
        device_ids=[0]
    return device,device_ids

def get_classes(json_path,label_number):
    files=os.listdir(json_path)
    file_list=[]
    for file in files:
        file_list.append(file)
    json_list=file_list[0:label_number]
    jsonfile_list=file_list[label_number:2*label_number] 
    json_file=json_list[0]
    img_file=jsonfile_list[0]
    data=json.load(open(json_path+json_file))
    img=cv2.imread(json_path+img_file+'/img.png')
    lbl, lbl_names = labelme.utils.labelme_shapes_to_label(img.shape, data['shapes'])
    class_name=[]
    for name in lbl_names:
        class_name.append(name)
    return class_name[1:]

def train_dataset_choose(out_folder,dataset_choose):
    dataset_path=out_folder+'dataset/'+dataset_choose
    return dataset_path

def get_name(n):
    col=n%10
    row=int(n/10)
    name='row='+str(row)+'_col='+str(col)
    return name
    
def Segmentation_train(Encoders,Encoder_weights,model_name,Activation,Epochs,batch_size:int,
                       dataset_choose,out_folder,json_path,label_number):
    dataset_path=train_dataset_choose(out_folder,dataset_choose)
    x_train_dir = os.path.join(dataset_path, 'train')
    y_train_dir = os.path.join(dataset_path, 'train_mask')
    x_valid_dir = os.path.join(dataset_path, 'test')
    y_valid_dir = os.path.join(dataset_path, 'test_mask')
    x_test_dir = os.path.join(dataset_path, 'val')
    y_test_dir = os.path.join(dataset_path, 'val_mask')
    loss = smp.utils.losses.DiceLoss()
        #记录指标
    metrics = [smp.utils.metrics.IoU(threshold=0.5),
               smp.utils.metrics.Accuracy(),
               smp.utils.metrics.Recall(),
               smp.utils.metrics.Precision(),
               smp.utils.metrics.Fscore(),]
    device,device_ids=cuda_is_available()
    Classes=get_classes(json_path,label_number)
    for i in range(len(Encoders)):
        model=Model_create(model_name=model_name,
                           Encoder_name=Encoders[i],
                           Encoder_weights=Encoder_weights,
                           CLASSES=Classes,
                           Activation=Activation)
        if torch.cuda.is_available():
            model=torch.nn.DataParallel(model,device_ids=device_ids)
        else:
            model=model
        preprocessing_fn = smp.encoders.get_preprocessing_fn(Encoders[i], Encoder_weights)
        train_dataset = Dataset(x_train_dir, 
                                y_train_dir, 
                                augmentation=get_training_augmentation(), 
                                preprocessing=get_preprocessing(preprocessing_fn),
                                classes=Classes,
                                )
        valid_dataset = Dataset(x_valid_dir, 
                                y_valid_dir, 
                                augmentation=get_validation_augmentation(), 
                                preprocessing=get_preprocessing(preprocessing_fn),
                                classes=Classes,
                                )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
        valid_loader = DataLoader(valid_dataset, batch_size=int(0.8*batch_size), shuffle=False)
        optimizer=torch.optim.Adam([dict(params=model.parameters(),lr=0.0001)])
        train_epoch=smp.utils.train.TrainEpoch(model,
                                               loss=loss,
                                               metrics=metrics,
                                               optimizer=optimizer,
                                               device=device,
                                               verbose=True,)
        valid_epoch=smp.utils.train.ValidEpoch(model,
                                               loss=loss,
                                               metrics=metrics,
                                               device=device,
                                               verbose=True,)
        max_score=0
        model_save_path=out_folder+'model/'+dataset_choose+'/'+model_name+'/'
        try:
            os.makedirs(model_save_path)
        except OSError:
            pass
        #train
        for j in range(0,Epochs):
            print('\nEpoch: {}'.format(j))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(model,model_save_path+Encoders[i]+'_best_model.pth')
                print('Model saved!')
            if j == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')
        print('The final results of :Unet++_',Encoders[i])
    print('All models have been trained and generated.')
    
def Segmentation_test(Activation,
                       dataset_choose,out_folder,json_path,label_number):
    dataset_path=train_dataset_choose(out_folder,dataset_choose)
    x_test_dir = os.path.join(dataset_path, 'val')
    y_test_dir = os.path.join(dataset_path, 'val_mask')
    ENCODER_WEIGHTS = 'imagenet'
    ENCODER_WEIGHTS_2='instagram'
    loss = smp.utils.losses.DiceLoss()
        #记录指标
    metrics = [smp.utils.metrics.IoU(threshold=0.5),
               smp.utils.metrics.Accuracy(),
               smp.utils.metrics.Recall(),
               smp.utils.metrics.Precision(),
               smp.utils.metrics.Fscore(),]
    device,device_ids=cuda_is_available()
    Classes=get_classes(json_path,label_number)
    check_path=out_folder+'model'
    print('Check for the existence of paths and models')
    model_dirname=[]
    model_filename=[]
    model_dirname_path=[]
    all_model=[]
    for parent, dirnames, filenames in os.walk(check_path):
        for dirname in dirnames:
            print("Model save path check:", parent)
            model_dirname.append(dirname)
            print("Model name:", dirname)
        for filename in filenames:
            print("Models Path check:", parent)
            model_filename.append(parent+'/'+filename)
            print("Encoder:", filename[:-15])
    print('Model test result')
    for parent,dirnames,filenames in os.walk(check_path):
        for dirname in dirnames:
            model_dirname.append(dirname)
            model_dirname_path.append(parent+'/'+dirname)
        for filename in filenames:
            encoder=filename[:-15]
            print('-'*60)
            print('Architecture',parent[len(check_path):])
            print('parent path:',parent,'\n','trained model name:',filename,'\n','used encoder:',encoder)
            state_model=parent+'/'+filename
            all_model.append(parent+'/'+filename)
            print('model path :',state_model)
            if encoder=='resnext101_32x16d':
                preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS_2)
            elif encoder == 'resnext101_32x8d':
                preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS_2)
            else:
                preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS)
            test_dataset = Dataset2(
                x_test_dir, 
                y_test_dir, 
                augmentation=get_validation_augmentation(), 
                preprocessing=get_preprocessing(preprocessing_fn),
                classes=Classes,
                )
            best_model = torch.load(state_model)
            test_dataloader = DataLoader(test_dataset)
            test_epoch = smp.utils.train.ValidEpoch(model=best_model,
                                                    loss=loss,
                                                    metrics=metrics,
                                                    device=device,
                                                    )
            logs = test_epoch.run(test_dataloader)
            print('-'*60)

def Segmentation_result(dataset_choose,out_folder,json_path,label_number):
    dataset_path=train_dataset_choose(out_folder,dataset_choose)
    device,device_ids=cuda_is_available()
    Classes=get_classes(json_path,label_number)
    cut_image_original=out_folder+'cut_image'
    cut_image_names=[]
    for parent,dirnames,filenames in os.walk(cut_image_original):
        for file in filenames:
            cut_image_names.append(file)
    model_dirname=[]
    model_dirname_path=[]
    model_path=[]
    all_model=[]
    metrics = [smp.utils.metrics.IoU(threshold=0.5),
               smp.utils.metrics.Accuracy(),
              smp.utils.metrics.Recall(),
              smp.utils.metrics.Precision(),
              smp.utils.metrics.Fscore(),]
    ENCODER_WEIGHTS = 'imagenet'
    ENCODER_WEIGHTS_2='instagram'
    ACTIVATION = 'sigmoid'
    models_path=out_folder+'model'
    models_path_tmp=out_folder+'model/'
    for parent,dirnames,filenames in os.walk(models_path):
        for dirname in dirnames:
            model_dirname.append(dirname)
            model_dirname_path.append(parent+'/'+dirname)
        for filename in filenames:
            encoder=filename[:-15]
            print('-'*60)
            if encoder=='resnext101_32x16d':
                preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS_2)
            elif encoder == 'resnext101_32x8d':
                preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS_2)
            elif encoder == 'resnext101_32x16d':
                preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS_2)
            else:
                preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS)
            architecture_name=parent[len(models_path_tmp):]
            print('Architecture',architecture_name)
            print('parent path:',parent,'\n','trained model name:',filename,'\n','used encoder:',encoder)
            state_model=parent+'/'+filename
            all_model.append(parent+'/'+filename)
            print('model path :',state_model)
            test_dataset_vis = Dataset(cut_image_original, 
                                       cut_image_original, 
                                       preprocessing=get_preprocessing(preprocessing_fn),
                                       classes=Classes
                                       )
            best_model = torch.load(state_model)
            try:
                os.makedirs(out_folder+'Segmentation_results/'+parent[len(models_path_tmp):]+'_'+encoder)
            except OSError:
                pass
            savepath=out_folder+'Segmentation_results/'+parent[len(models_path_tmp):]+'_'+encoder+'/'
            for i in range(len(cut_image_names)):
                name=get_name(i)
                image_vis=test_dataset_vis[i][0].astype('uint8')
                image,mask=test_dataset_vis[i]
                mask=mask.squeeze()
                x_tensor=torch.from_numpy(image).to(device).unsqueeze(0)
                pr_mask=best_model.module.predict(x_tensor)
                pr_mask=pr_mask.squeeze().cpu().numpy().round()
                plt.figure(figsize=(5,5),frameon=False)
                plt.axis('off')
                plt.tight_layout(pad = 0)
                if np.asarray(pr_mask).shape==(800,800):
                    plt.imshow(-pr_mask,cmap=plt.cm.gray)
                    plt.savefig(savepath+name+'.png',dpi=160)
                else:
                    plt.imshow(-pr_mask[0],cmap=plt.cm.gray)
                    plt.savefig(savepath+name+'.png',dpi=160)
            print(architecture_name,'cut_image results are saved!')
            #paste them
            past_image_list=os.listdir(savepath)
            target=Image.new('RGB',(8000,8000))
            paste_size=800
            left_num_p=0
            top_num_p=0
            img=[]
            for n in past_image_list:
                img.append(Image.open(savepath+n))
            for i in range(1,11):
                left_num_p=0
                for j in range(1,11):
                    a=paste_size*left_num_p #zuo
                    b=paste_size*top_num_p #shang
                    c=paste_size*(left_num_p+1) #you
                    d=paste_size*(top_num_p+1) #xia
                    target.paste(img[10*(i-1)+j-1], (a, b, c, d))
                    left_num_p+=1
                top_num_p+=1
            target.save(savepath[:-1]+'_result.png')

'''
 4. Segmentation train process
Segmentation_train(Encoders=Encoders,
                   Encoder_weights=Encoder_weights,
                   model_name=model_name,
                   Activation=Activation,
                   Epochs=Epoch,
                   batch_size=train_batch_size,
                   dataset_choose=dataset_choose,
                   out_folder=out_folder,
                   json_path=json_path,
                   label_number=label_number)

 5. Segmentation test process from trained models
Segmentation_test(Activation=Activation,
                  dataset_choose=dataset_choose,
                  out_folder=out_folder,
                  json_path=json_path,
                  label_number=label_number)

# 6. Segmentation result on visualization and save result
Segmentation_result(dataset_choose=dataset_choose,
                    out_folder=out_folder,
                    json_path=json_path,
                    label_number=label_number)

'''
