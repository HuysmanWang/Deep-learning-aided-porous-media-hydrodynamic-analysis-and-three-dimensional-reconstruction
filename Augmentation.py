#Augmentation 数据增强

from PIL import Image
import random
import os
from torchvision import transforms
import torchvision.transforms.functional as tf
from Image_preprocess import *
#label_path作为ground truth

####################需要提供的数据####################
out_folder = 'H:/tmp/DEEP_WATERROCK_CODE/codetest/'   #数据存储路径
label_number=5 #想要标记图片的数量
if_labelme=0 #0不打开，1打开labelme
file_path = 'H:/tmp/DEEP_WATERROCK_CODE/test_image/test_1.png'  #要分割的图片地址
json_path='H:/tmp/json/'  #labelme之后翻译得到的地址,路径不要出现中文
label_number=5 #想要标记图片的数量
train_num=20 #train_num*label_number=训练文件总数
test_num=5 #test_num*label_number=测试文件总数
val_num=5 #val_num*label_number=验证文件总数
#####################################################

def make_save_path(dataset_path,mask_files_list):
    for i in range(len(mask_files_list)):
        try:
            os.makedirs(dataset_path+mask_files_list[i])
        except OSError:
            pass

def make_dataset_dirs(dataset_path):
    try:
        os.makedirs(dataset_path+'/train')
        os.makedirs(dataset_path+'/train_mask')
        os.makedirs(dataset_path+'/test')
        os.makedirs(dataset_path+'/test_mask')
        os.makedirs(dataset_path+'/val')
        os.makedirs(dataset_path+'/val_mask')
    except OSError:
        pass

class Augmentation:
    def __init__(self):
        pass
    def rotate(self,image,mask,angle=None):
        if angle == None:
            angle = transforms.RandomRotation.get_params([-180, 180]) # -180~180随机选一个角度旋转
        if isinstance(angle,list):
            angle = random.choice(angle)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask
    def flip(self,image,mask): #水平翻转和垂直翻转
        if random.random()>0.5:
            image = tf.hflip(image)
            mask = tf.hflip(mask)
        if random.random()<0.5:
            image = tf.vflip(image)
            mask = tf.vflip(mask)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask
    def adjustContrast(self,image,mask):
        factor = transforms.RandomRotation.get_params([0,10])   #这里调增广后的数据的对比度
        image = tf.adjust_contrast(image,factor)
        #mask = tf.adjust_contrast(mask,factor)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image,mask
    def adjustBrightness(self,image,mask):
        factor = transforms.RandomRotation.get_params([1, 2])  #这里调增广后的数据亮度
        image = tf.adjust_brightness(image, factor)
        #mask = tf.adjust_contrast(mask, factor)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask
    def adjustSaturation(self,image,mask): #调整饱和度
        factor = transforms.RandomRotation.get_params([1, 2])  # 这里调增广后的数据亮度
        image = tf.adjust_saturation(image, factor)
        #mask = tf.adjust_saturation(mask, factor)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask

#train data create
def augmentationData_train(image_path,mask_path,option=[1,2,3,4,5],save_dir=None,multiple=70):
    aug_image_savedDir = os.path.join(save_dir,'train')
    aug_mask_savedDir = os.path.join(save_dir, 'train_mask')
    aug = Augmentation()
    res_image= os.walk(image_path)
    images = []
    masks = []
    for root,dirs,files in res_image:
        for f in files:
            images.append(os.path.join(root,f))
    res_mask = os.walk(mask_path)
    for root,dirs,files in res_mask:
        for f in files:
            masks.append(os.path.join(root,f))
    datas = list(zip(images,masks))
    num = len(datas)
    for epoch in range(int(multiple/5)): #生成100组数据用于训练，原图用于最终测试
        for (image_path,mask_path) in datas:
            image = Image.open(image_path)
            mask = Image.open(mask_path)
            if 1 in option:
                num+=1
                image_tensor, mask_tensor = aug.rotate(image, mask)
                image_rotate = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'train', str(num) + '_rotate.png'))
                mask_rotate = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'train_mask', str(num) + '_rotate.png'))
            if 2 in option:
                num+=1
                image_tensor, mask_tensor = aug.flip(image, mask)
                image_filp = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir,'train',str(num)+'_filp.png'))
                mask_filp = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir,'train_mask',str(num)+'_filp.png'))
            if 3 in option:
                num+=1
                image_tensor, mask_tensor = aug.adjustContrast(image, mask)
                image_Contrast = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'train', str(num) + '_Contrast.png'))
                mask_Contrast = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'train_mask', str(num) + '_Contrast.png'))
            if 4 in option:
                num+=1
                image_tensor, mask_tensor = aug.adjustBrightness(image, mask)
                image_Brightness = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'train', str(num) + '_Brightness.png'))
                mask_Brightness = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'train_mask', str(num) + '_Brightness.png'))
            if 5 in option:
                num+=1
                image_tensor, mask_tensor = aug.adjustSaturation(image, mask)
                image_Saturation = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'train', str(num) + '_Saturation.png'))
                mask_Saturation = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'train_mask', str(num) + '_Saturation.png'))
            
#test data create
def augmentationData_test(image_path,mask_path,option=[1,2,3,4,5],save_dir=None,multiple=20):
    aug_image_savedDir = os.path.join(save_dir,'test')
    aug_mask_savedDir = os.path.join(save_dir, 'test_mask')
    aug = Augmentation()
    res_image= os.walk(image_path)
    images = []
    masks = []
    for root,dirs,files in res_image:
        for f in files:
            images.append(os.path.join(root,f))
    res_mask = os.walk(mask_path)
    for root,dirs,files in res_mask:
        for f in files:
            masks.append(os.path.join(root,f))
    datas = list(zip(images,masks))
    num = len(datas)
    for epoch in range(int(multiple/5)): #生成100组数据用于test，原图用于最终测试
        for (image_path,mask_path) in datas:
            image = Image.open(image_path)
            mask = Image.open(mask_path)
            if 1 in option:
                num+=1
                image_tensor, mask_tensor = aug.rotate(image, mask)
                image_rotate = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'test', str(num) + '_rotate.png'))
                mask_rotate = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'test_mask', str(num) + '_rotate.png'))
            if 2 in option:
                num+=1
                image_tensor, mask_tensor = aug.flip(image, mask)
                image_filp = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir,'test',str(num)+'_filp.png'))
                mask_filp = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir,'test_mask',str(num)+'_filp.png'))
            if 3 in option:
                num+=1
                image_tensor, mask_tensor = aug.adjustContrast(image, mask)
                image_Contrast = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'test', str(num) + '_Contrast.png'))
                mask_Contrast = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'test_mask', str(num) + '_Contrast.png'))
            if 4 in option:
                num+=1
                image_tensor, mask_tensor = aug.adjustBrightness(image, mask)
                image_Brightness = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'test', str(num) + '_Brightness.png'))
                mask_Brightness = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'test_mask', str(num) + '_Brightness.png'))
            if 5 in option:
                num+=1
                image_tensor, mask_tensor = aug.adjustSaturation(image, mask)
                image_Saturation = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'test', str(num) + '_Saturation.png'))
                mask_Saturation = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'test_mask', str(num) + '_Saturation.png'))
                
#validation data create
def augmentationData_validation(image_path,mask_path,option=[1,2,3,4,5],save_dir=None,multiple=10):
    aug_image_savedDir = os.path.join(save_dir,'val')
    aug_mask_savedDir = os.path.join(save_dir, 'val_mask')
    aug = Augmentation()
    res_image= os.walk(image_path)
    images = []
    masks = []
    for root,dirs,files in res_image:
        for f in files:
            images.append(os.path.join(root,f))
    res_mask = os.walk(mask_path)
    for root,dirs,files in res_mask:
        for f in files:
            masks.append(os.path.join(root,f))
    datas = list(zip(images,masks))
    num = len(datas)
    for epoch in range(int(multiple/5)): #生成100组数据用于validation，原图用于最终测试
        for (image_path,mask_path) in datas:
            image = Image.open(image_path)
            mask = Image.open(mask_path)
            if 1 in option:
                num+=1
                image_tensor, mask_tensor = aug.rotate(image, mask)
                image_rotate = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'val', str(num) + '_rotate.png'))
                mask_rotate = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'val_mask', str(num) + '_rotate.png'))
            if 2 in option:
                num+=1
                image_tensor, mask_tensor = aug.flip(image, mask)
                image_filp = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir,'val',str(num)+'_filp.png'))
                mask_filp = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir,'val_mask',str(num)+'_filp.png'))
            if 3 in option:
                num+=1
                image_tensor, mask_tensor = aug.adjustContrast(image, mask)
                image_Contrast = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'val', str(num) + '_Contrast.png'))
                mask_Contrast = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'val_mask', str(num) + '_Contrast.png'))
            if 4 in option:
                num+=1
                image_tensor, mask_tensor = aug.adjustBrightness(image, mask)
                image_Brightness = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'val', str(num) + '_Brightness.png'))
                mask_Brightness = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'val_mask', str(num) + '_Brightness.png'))
            if 5 in option:
                num+=1
                image_tensor, mask_tensor = aug.adjustSaturation(image, mask)
                image_Saturation = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'val', str(num) + '_Saturation.png'))
                mask_Saturation = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'val_mask', str(num) + '_Saturation.png'))

def augmentation_dataset_create(gt_path,mask_files_list,save_path_list,train_num=70,test_num=20,val_num=10): #num * number = 文件数 num=5的倍数
    for i in range(len(save_path_list)):
        augmentationData_train(image_path=gt_path,
                               mask_path=mask_files_list[i],
                               save_dir=save_path_list[i],
                               multiple=train_num)
        augmentationData_test(image_path=gt_path,
                              mask_path=mask_files_list[i],
                              save_dir=save_path_list[i],
                              multiple=test_num)
        augmentationData_validation(image_path=gt_path,
                                    mask_path=mask_files_list[i],
                                    save_dir=save_path_list[i],
                                    multiple=val_num)
        
def Augmentation_dataset(file_path,out_folder,json_path,label_number,
                         train_num,test_num,val_num):
    try:
        os.makedirs(out_folder+'dataset')
    except OSError:
        pass
    dataset_path=out_folder+'dataset/'
    path=train_set_create(file_path,out_folder)
    gt_path=path['label_path']
    mask_path=json2dataset(json_path,label_number,out_folder)
    mask_files=os.listdir(mask_path)
    mask_files_list=[]     #存储mask的路径
    mask_path_list=[]
    save_path_list=[]    #存储dataset save的路径
    for file in mask_files:
        mask_files_list.append(file)
        mask_path_list.append(mask_path+file)
    for file in mask_files:
        save_path_list.append(dataset_path+file)
    make_save_path(dataset_path,mask_files_list)
    for i in range(len(save_path_list)):
        make_dataset_dirs(save_path_list[i])
    augmentation_dataset_create(gt_path,mask_path_list,save_path_list,
                            train_num=train_num,test_num=test_num,val_num=val_num)
    return save_path_list

'''
 3. Augmentation_dataset for few-shot learning
save_path_list=Augmentation_dataset(file_path,out_folder,json_path,label_number,train_num,test_num,val_num)
'''
