#LBM 2D Permeability Analysis D2Q9

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import os
from pylab import meshgrid,  arange, streamplot, show
import time

####################需要提供的数据####################
#Image_path
device='cuda'
out_folder='H:/tmp/DEEP_WATERROCK_CODE/codetest/'
path1='H:/清华大学/论文/论文《DEEP FLOW》王明阳/深度分割重建计算/SEGMENTATION_RESULTS/results_image/formerwork/1_1.png'
path2='H:/清华大学/论文/论文《DEEP FLOW》王明阳/深度分割重建计算/SEGMENTATION_RESULTS/results_image/planB/O_result/Unet++_densenet201/1.png'
#####################################################

def get_img_obstacle(img_path):
    num_0=0
    num_1=0
    obstacle_0=[]
    obstacle_1=[]
    img=Image.open(img_path)
    data=np.array(img)/255.0
    data[data!=1]=0
    for x in range(0,data.shape[0]):
        for y in range(0,data.shape[1]):
            if data[x][y].all()==0:
                num_0+=1
                obstacle_0.append((x+1)*(y+1))
            else:
                num_1+=1
                obstacle_1.append((x+1)*(y+1))
    return obstacle_0,obstacle_1,data

def to_tensor_gpu(x):
    return torch.tensor(x).to(device)

#solver适用于（3，800,800）
def lbm_solver(img_path):
    #开始计时
    time0=time.time()
    #初值定义
    omega=1.0
    density=1.0
    t1=4/9
    t2=1/9
    t3=1/36
    c_squ=1/3
    avu=1
    prevavu=1
    ts=0
    deltaU=1e-7
    cxs = np.array([1, 1, 0,-1,-1,-1, 0, 1, 0])
    cys = np.array([0, 1, 1, 1, 0,-1,-1,-1, 0])
    weights = np.array([1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36,4/9]) 
    NL=9
    idxs=np.arange(9)
    device='cuda'
    #读取数据
    img_path=img_path
    obstacle,active_nodes,BOUND=get_img_obstacle(img_path)
    BOUND=BOUND.astype('bool')
    #print('number of obstacle:',len(obstacle),'\n','number of active nodes:',len(active_nodes))
    porosity=len(obstacle)/(len(obstacle)+len(active_nodes))
    print('porosity:',porosity)
    print('Calc nodes number:',(len(obstacle)+len(active_nodes)))
    BOUND=BOUND[:,:]
    nx=BOUND.shape[0]
    ny=BOUND.shape[1]
    F=np.tile(density/9,[nx,ny,9]) #F=repmat(density/9,[nx ny 9])
    FEQ=F    
    #cuda加速
    F1=to_tensor_gpu(F)
    FEQ1=to_tensor_gpu(FEQ)
    cxs1=to_tensor_gpu(cxs)
    cys1=to_tensor_gpu(cys)
    weights1=to_tensor_gpu(weights)
    idxs1=to_tensor_gpu(idxs)
    avu1=to_tensor_gpu(avu)
    prevavu1=to_tensor_gpu(prevavu)
    ts1=to_tensor_gpu(ts)
    #开始迭代计算
    while ts1<4000 and 1e-10<torch.abs((prevavu1-avu1)/avu1) or ts1<100:
        for i,cx,cy in zip(idxs,cxs,cys):
            F1[:,:,i]=torch.roll(F1[:,:,i],(cx,cy),(1,0))
        boundary_F1=F1[BOUND,:]
        boundary_F1= boundary_F1[:,[4,5,6,7,0,1,2,3,8]]
        DENSITY=F1.sum(axis=2)
        UX=(F1*cxs1).sum(axis=2)/DENSITY
        UY=(F1*cys1).sum(axis=2)/DENSITY
        UX[0,0:ny]=UX[0,0:ny]+deltaU
        UX[BOUND]=0
        UY[BOUND]=0
        DENSITY[BOUND]=0
        U_SQU=UX**2+UY**2
        for i,cx,cy,w in zip(idxs,cxs,cys,weights):
            FEQ1[:,:,i]=DENSITY*w*(1+3*(cx*UX+cy*UY)+9*(cx*UX+cy*UY)**2/2-3*(UX**2+UY**2)/2)
        F1=omega*FEQ1+(1-omega)*F1
        F1[BOUND,:] =boundary_F1    
        prevavu1=avu1
        avu1=((UX.sum(axis=0)).sum(axis=0))/len(active_nodes)
        ts1+=1
        time2=time.time()
    time_all=time2-time0
    F1=F1.cpu().numpy()
    UX=UX.cpu().numpy()
    UY=UY.cpu().numpy()
    U_SQU=U_SQU.cpu().numpy()
    x, y = meshgrid(arange(0,ny), arange(0,nx))
    result={'mesh_x':x,
            'mesh_y':y,
            'Velocity_x':UX,
            'Velocity_y':UY,
            'Velocity_square':U_SQU,
            'Calc_time':time_all,
            'Geometry':BOUND,
            'porosity':porosity}
    return result

#solver_2适用于（800,800）
def lbm_solver_2(img_path):
    #开始计时
    time0=time.time()
    #初值定义
    omega=1.0
    density=1.0
    t1=4/9
    t2=1/9
    t3=1/36
    c_squ=1/3
    avu=1
    prevavu=1
    ts=0
    deltaU=1e-7
    cxs = np.array([1, 1, 0,-1,-1,-1, 0, 1, 0])
    cys = np.array([0, 1, 1, 1, 0,-1,-1,-1, 0])
    weights = np.array([1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36,4/9]) 
    NL=9
    idxs=np.arange(9)
    device='cuda'
    #读取数据
    img_path=img_path
    obstacle,active_nodes,BOUND=get_img_obstacle(img_path)
    BOUND=BOUND.astype('bool')
    #print('number of obstacle:',len(obstacle),'\n','number of active nodes:',len(active_nodes))
    porosity=len(obstacle)/(len(obstacle)+len(active_nodes))
    print('porosity:',porosity)
    print('Calc nodes number:',(len(obstacle)+len(active_nodes)))
    BOUND=BOUND[:,:,0]
    nx=BOUND.shape[0]
    ny=BOUND.shape[1]
    F=np.tile(density/9,[nx,ny,9]) #F=repmat(density/9,[nx ny 9])
    FEQ=F    
    #cuda加速
    F1=to_tensor_gpu(F)
    FEQ1=to_tensor_gpu(FEQ)
    cxs1=to_tensor_gpu(cxs)
    cys1=to_tensor_gpu(cys)
    weights1=to_tensor_gpu(weights)
    idxs1=to_tensor_gpu(idxs)
    avu1=to_tensor_gpu(avu)
    prevavu1=to_tensor_gpu(prevavu)
    ts1=to_tensor_gpu(ts)
    #开始迭代计算
    while ts1<4000 and 1e-10<torch.abs((prevavu1-avu1)/avu1) or ts1<100:
        for i,cx,cy in zip(idxs,cxs,cys):
            F1[:,:,i]=torch.roll(F1[:,:,i],(cx,cy),(1,0))
        boundary_F1=F1[BOUND,:]
        boundary_F1= boundary_F1[:,[4,5,6,7,0,1,2,3,8]]
        DENSITY=F1.sum(axis=2)
        UX=(F1*cxs1).sum(axis=2)/DENSITY
        UY=(F1*cys1).sum(axis=2)/DENSITY
        UX[0,0:ny]=UX[0,0:ny]+deltaU
        UX[BOUND]=0
        UY[BOUND]=0
        DENSITY[BOUND]=0
        U_SQU=UX**2+UY**2
        for i,cx,cy,w in zip(idxs,cxs,cys,weights):
            FEQ1[:,:,i]=DENSITY*w*(1+3*(cx*UX+cy*UY)+9*(cx*UX+cy*UY)**2/2-3*(UX**2+UY**2)/2)
        F1=omega*FEQ1+(1-omega)*F1
        F1[BOUND,:] =boundary_F1    
        prevavu1=avu1
        avu1=((UX.sum(axis=0)).sum(axis=0))/len(active_nodes)
        ts1+=1
        time2=time.time()
    time_all=time2-time0
    F1=F1.cpu().numpy()
    UX=UX.cpu().numpy()
    UY=UY.cpu().numpy()
    U_SQU=U_SQU.cpu().numpy()
    x, y = meshgrid(arange(0,ny), arange(0,nx))
    result={'mesh_x':x,
            'mesh_y':y,
            'Velocity_x':UX,
            'Velocity_y':UY,
            'Velocity_square':U_SQU,
            'Calc_time':time_all,
            'Geometry':BOUND,
            'porosity':porosity}
    return result

def collect_path(**path):
    path_all=[]
    name_all=[]
    for i,(name,path) in enumerate(path.items()):
        path_all.append(path)
        name_all.append(name)
    return path_all,name_all

def LBM_2D_Analysis(path_all,out_folder):
    device='cuda'
    try:
        os.makedirs(out_folder+'LBM2D')
    except OSError:
        pass
    save_path=out_folder+'LBM2D/'
    
    result_all=[]
    X=[]
    Y=[]
    UX=[]
    UY=[]
    U_SQU=[]
    Time=[]
    GEO=[]
    Pore=[]
    #calc all
    for i in range(len(path_all)):
        img=np.asarray(Image.open(path_all[i]))
        if len(img.shape)==2:
            result=lbm_solver(path_all[i])
        else:
            result=lbm_solver_2(path_all[i])
        result_all.append(result)
    #save all array
    for j in range(len(result_all)):
        result=result_all[j]
        X.append(result['mesh_x'])
        Y.append(result['mesh_y'])
        UX.append(result['Velocity_x'])
        UY.append(result['Velocity_y'])
        U_SQU.append(result['Velocity_square'])
        Time.append(result['Calc_time'])
        GEO.append(result['Geometry'])
        Pore.append(result['porosity'])
    #plot
    font = {'family' : 'Arial',
            'color'  : 'black',
            'weight' : 'normal',
            'size'   : 30,
            }
    #steam plot
    try:
        os.makedirs(save_path+'stream_plot')
    except OSError:
        pass
    stream_plot_path=save_path+'stream_plot/'
    for i in range(len(X)):
        plt.figure(figsize=(8,5),dpi=120)
        plt.imshow(~GEO[i],cmap=plt.cm.gray)
        color = 2 * (np.hypot(UX[i], UY[i]))
        plt.axis('off')
        streamplot(X[i],Y[i],UX[i],UY[i],
               density=2.5,
               color=color,
               cmap=plt.cm.jet,
               arrowstyle='fancy',
               arrowsize=0.8,
               minlength=0.1,
               maxlength=10,
              )
        h=plt.colorbar(fraction=0.04,pad=0.03,shrink=0.9)
        h.ax.tick_params(labelsize=15)
        plt.savefig(stream_plot_path+name_all[i]+'.png',
                   bbox_inches='tight')
    #UX plot
    try:
        os.makedirs(save_path+'UX_plot')
    except OSError:
        pass
    UX_plot_path=save_path+'UX_plot/'
    for i in range(len(X)):
        plt.figure(figsize=(7,7),dpi=160)
        plt.contourf(UX[i],cmap='jet')
        h=plt.colorbar(fraction=0.04,pad=0.03,shrink=0.9)
        h.ax.tick_params(labelsize=18)
        plt.axis('off')
        plt.savefig(UX_plot_path+name_all[i]+'.png',
                    bbox_inches='tight')
    #UY plot
    try:
        os.makedirs(save_path+'UY_plot')
    except OSError:
        pass
    UY_plot_path=save_path+'UY_plot/'
    for i in range(len(X)):
        plt.figure(figsize=(7,7),dpi=160)
        plt.contourf(UY[i],cmap='jet')
        h=plt.colorbar(fraction=0.04,pad=0.03,shrink=0.9)
        h.ax.tick_params(labelsize=18)
        plt.axis('off')
        plt.savefig(UY_plot_path+name_all[i]+'.png',
                    bbox_inches='tight')
    #U_SQU plot
    try:
        os.makedirs(save_path+'U_SQU_plot')
    except OSError:
        pass
    U_SQU_plot_path=save_path+'U_SQU_plot/'
    for i in range(len(X)):
        plt.figure(figsize=(7,7),dpi=160)
        plt.contourf(U_SQU[i],cmap='jet')
        h=plt.colorbar(fraction=0.04,pad=0.03,shrink=0.9)
        h.ax.tick_params(labelsize=18)
        plt.axis('off')
        plt.savefig(U_SQU_plot_path+name_all[i]+'.png',
                    bbox_inches='tight')
    #Time log
    for i in range(len(Time)):
        f=open(save_path+'calc_time_log.txt','a')
        f.write(name_all[i]+'    {:2.4f} s'.format(Time[i]))
        f.write('\n')
        f.close()
    print('Calculation time per model saved in calc_time_log.txt')
    #Porosity log
    for i in range(len(X)):
        f=open(save_path+'porosity_log.txt','a')
        f.write(name_all[i]+'    {:2.4f}'.format(Pore[i]))
        f.write('\n')
        f.close()
        
        
path_all,name_all=collect_path(fw=path1,ow=path2)
'''
 7. LBM_2D_Analysis
LBM_2D_Analysis(path_all,out_folder)
'''
