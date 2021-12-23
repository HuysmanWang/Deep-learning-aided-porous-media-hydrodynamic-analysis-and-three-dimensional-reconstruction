#Any image input no matter what scale it belongs, should through following steps that used for Image segmentation task

#Including two parts: 
# 1. Segmentation training set prepare 8000->800->choose 5 label->label me
# 2. Segmentation model predict

from PIL import Image
import cv2
import os
import random
import json
import labelme
import numpy as np

####################需要提供的数据####################
out_folder = 'H:/tmp/DEEP_WATERROCK_CODE/codetest/'   #数据存储路径
label_number=5 #想要标记图片的数量
if_labelme=0 #0不打开，1打开labelme
file_path = 'H:/tmp/DEEP_WATERROCK_CODE/test_image/test_1.png'  #要分割的图片地址
json_path='H:/tmp/json/'  #labelme之后翻译得到的地址,路径不要出现中文
#####################################################

def normalize_image_size(input_img_path,out_path):
    img=cv2.imread(input_img_path)
    img=cv2.resize(img,(8000,8000))
    new_path=out_path+'normalize_image.png'
    cv2.imwrite(new_path,img)
    return new_path

def cut_image(image,width_out,height_out,path):
    width,height=image.size
    width_num=int(width/width_out)
    height_num=int(height/height_out)
    for j in range(width_num):
        for i in range(height_num):
            box = (width_out*i,height_out*j, width_out* (i + 1), height_out* (j + 1))
            region = image.crop(box)
            region.save(path+'width_num={}_height_num={}.png'.format(j, i))
    return path

def random_choose_image(number:int,original_path):
    img_list=[]
    files=os.listdir(original_path)
    for file in files:
        img_list.append(original_path+file)
    img_choose=random.sample(img_list,number)
    return img_choose

def choose_image_resize(img_choose,label_path):
    n=1
    for img in img_choose:
        img=cv2.imread(img)
        img=cv2.resize(img,(800,800))
        cv2.imwrite(label_path+str(n)+'.png',img)
        n+=1
    return label_path

def train_set_create(file_path,out_folder):
    image_path=normalize_image_size(file_path,out_folder)
    image=Image.open(image_path)          #读取图片
    image_height,image_width=image.size
    height=image_height/10
    width=image_width/10
    cut_image_save_path=out_folder+'cut_image/'
    label_image_path=out_folder+'label_image/'
    try:
        os.makedirs(cut_image_save_path)
    except OSError:
        pass
    try:
        os.makedirs(label_image_path)
    except OSError:
        pass
    image_cut_path = cut_image(image,width,height,cut_image_save_path)        #分割图片
    image_cut_choose=random_choose_image(label_number,image_cut_path)     #随机挑选需要label的图片
    image_label_path=choose_image_resize(image_cut_choose,label_image_path)     #resize为（800,800）用来标记的图片集
    path={'cut_path':image_cut_path,'label_path':image_label_path}
    return path

def label_me(num=1):
    if num==1:
        os.system('labelme')
    elif num==0:
        pass
    #利用labelme标记需要训练的数据集
def json2dataset(json_path,label_number,out_folder):
    files=os.listdir(json_path)
    file_list=[]
    for file in files:
        file_list.append(file)
    json_list=file_list[0:label_number]
    jsonfile_list=file_list[label_number:2*label_number]
    for n in range(label_number):
        json_file=json_list[n]
        img_file=jsonfile_list[n]
        data=json.load(open(json_path+json_file))
        img=cv2.imread(json_path+img_file+'/img.png')
        lbl, lbl_names = labelme.utils.labelme_shapes_to_label(img.shape, data['shapes'])
        mask=[]
        class_id=[]
        class_name=[]
        for name in lbl_names:
            class_name.append(name)
        for i in range(1,len(lbl_names)):
            mask.append((lbl==i).astype(np.uint8))
            class_id.append(lbl_names)
        mask=np.asarray(mask,np.uint8)
        mask_path=out_folder+'mask_image/'
        for j in range(0,len(class_name)-1):
            try:
                os.makedirs(mask_path+str(class_name[j+1]))
            except OSError:
                pass
            cv2.imwrite(mask_path+str(class_name[j+1])+'/'+str(n+1)+'.png',mask[j,:,:])
    return mask_path

def model_predict_set_create(file_path,out_folder):
    image_path=normalize_image_size(file_path,out_folder)
    image=Image.open(image_path)          #读取图片
    image_height,image_width=image.size
    height=image_height/10
    width=image_width/10
    cut_image_save_path=out_folder+'cut_image/'
    try:
        os.makedirs(cut_image_save_path)
    except OSError:
        pass
    image_cut_path = cut_image(image,width,height,cut_image_save_path)        #分割图片,100张
    return image_cut_path

'''
 1. Segmentation training set prepare
path=train_set_create(file_path,out_folder)
label_me(if_labelme)
cmd->labelme_json_to_dataset
mask_path,Classes=json2dataset(json_path,label_number,out_folder)

 2. Segmentation model predict
model_predict_set_create(file_path,out_folder)
'''
