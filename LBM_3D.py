##LBM 3D Permeability Analysis D3Q19 for reconstruction structure(256,256)

import cv2
import os
from PIL import Image
import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pylab import meshgrid,  arange, streamplot, show
from matplotlib.colors import Normalize

####################需要提供的数据####################
#Image_path
device='cuda'
out_folder='H:/tmp/DEEP_WATERROCK_CODE/codetest/'
path='H:/清华大学/论文/论文《DEEP FLOW》王明阳/深度分割重建计算/study_gan_6/result'
#####################################################

def get_obstacle(img_path):
    img_list=os.listdir(img_path)
    array=[]
    for img in img_list:
        gray_img=Image.open(img_path+img)
        layer=np.asarray(gray_img)
        array.append(layer)
    arr=np.asarray(array)
    GEO=arr[3:259,:,:,0]
    GEO_norm=GEO/255.
    num_0=0
    num_1=1
    obstacle_0=[]
    obstacle_1=[]
    GEO_norm[GEO_norm!=0]=1
    for i in range(0,GEO_norm.shape[0]):
        for j in range(0,GEO_norm.shape[1]):
            for k in range(0,GEO_norm.shape[2]):
                if GEO_norm[i][j][k].any()==0:
                    num_0+=1
                    obstacle_0.append((i+1)*(j+1)*(k+1)) #pore
                else:
                    num_1+=1
                    obstacle_1.append((i+1)*(j+1)*(k+1)) #obstacle
    return num_0,num_1,GEO_norm

def to_tensor_gpu(x):
    device='cuda'
    return torch.tensor(x).to(device)

def lbm_solver_3D(img_path):
    #开始计时
    time0=time.time()
    #初值定义
    omega=1.0
    density=1.0
    t1=1/3
    t2=1/18
    t3=1/36
    c_squ=1/3
    avu=1
    prevavu=1
    ts=0
    deltaU=1e-7
    cxs=np.array([1,0,0,-1,0,0,1,1,-1,-1,1,1,-1,-1,0,0,0,0,0])
    cys=np.array([0,1,0,0,-1,0,1,-1,1,-1,0,0,0,0,1,1,-1,-1,0])
    czs=np.array([0,0,1,0,0,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0])
    weights=np.array([t2,t2,t2,t2,t2,t2,
                 t3,t3,t3,t3,t3,t3,t3,t3,t3,t3,t3,t3,
                 t1])
    NL=19
    idxs=np.arange(19)
    device='cuda'
    #读取数据
    pore_num,obstacle_num,BOUND=get_obstacle(img_path)
    BOUND=BOUND.astype('bool')
    print('pore number:',pore_num,'\n','obstacle number:',obstacle_num,'\n','porosity:',pore_num/(pore_num+obstacle_num))
    porosity=pore_num/(pore_num+obstacle_num)
    BOUND=BOUND[:,:,:]
    nx=BOUND.shape[0]
    ny=BOUND.shape[1]
    nz=BOUND.shape[2]
    F=np.tile(density/19,[nx,ny,nz,19]) #F=repmat(density/9,[nx ny 9])
    FEQ=F
    #cuda加速
    F=to_tensor_gpu(F)
    FEQ=to_tensor_gpu(FEQ)
    cxs=to_tensor_gpu(cxs)
    cys=to_tensor_gpu(cys)
    czs=to_tensor_gpu(czs)
    weights=to_tensor_gpu(weights)
    idxs=to_tensor_gpu(idxs)
    avu=to_tensor_gpu(avu)
    prevavu=to_tensor_gpu(prevavu)
    ts=to_tensor_gpu(ts)
    #开始迭代计算
    while ts<8000 and 1e-7<torch.abs((prevavu-avu)/avu) or ts<100:
        for i,cx,cy,cz in zip(idxs,cxs,cys,czs):
            F[:,:,:,i]=torch.roll(F[:,:,:,i],(cx,cy,cz),(2,1,0))
        boundary_F=F[BOUND,:]
        boundary_F= boundary_F[:,[3,4,5,0,1,2,9,8,7,6,13,12,11,10,17,16,15,14,18]]
        DENSITY=F.sum(axis=3)
        UX=(F*cxs).sum(axis=3)/DENSITY
        UY=(F*cys).sum(axis=3)/DENSITY
        UZ=(F*czs).sum(axis=3)/DENSITY
        UX[0,:,:]=UX[0,:,:]+deltaU
        
        UX[BOUND,:]=0
        UY[BOUND,:]=0
        UZ[BOUND,:]=0
        DENSITY[BOUND,:]=0
        U_SQU=UX**2+UY**2+UZ**2
        for i,cx,cy,cz,w in zip(idxs,cxs,cys,czs,weights):
            FEQ[:,:,:,i]=DENSITY*w*(1+3*(cx*UX+cy*UY+cz*UZ)+9*(cx*UX+cy*UY+cz*UZ)**2/2-3*(UX**2+UY**2+UZ**2)/2)
        F=omega*FEQ+(1-omega)*F
        F[BOUND,:] =boundary_F    
        prevavu=avu
        avu=(((UX.sum(axis=0)).sum(axis=0)).sum(axis=0))/obstacle_num
        ts+=1
    time2=time.time()
    time_all=time2-time0
    F=F.cpu().numpy()
    UX=UX.cpu().numpy()
    UY=UY.cpu().numpy()
    UZ=UZ.cpu().numpy()
    U_SQU=U_SQU.cpu().numpy()
    x, y ,z= meshgrid(arange(0,ny), arange(0,nx),arange(0,nz))
    result={'mesh_x':x,
            'mesh_y':y,
            'mesh_z':z,
            'Velocity_x':UX,
            'Velocity_y':UY,
            'Velocity_z':UZ,
            'Velocity_square':U_SQU,
            'Calc_time':time_all,
            'Geometry':BOUND,
            'Porosity':porosity,
            'Force':F
            }
    return result

def LBM_3D_Analysis(path,out_folder):
    try:
        os.makedirs(out_folder+'LBM_3D')
    except OSError:
        pass
    save_path=out_folder+'LBM_3D/'
    
    result=lbm_solver_3D(path)
    
    F=result['Force']
    X=result['mesh_x']
    Y=result['mesh_y']
    Z=result['mesh_z']
    UX=result['Velocity_x']
    UY=result['Velocity_y']
    UZ=result['Velocity_z']
    U_SQU=result['Velocity_square']
    Time=result['Calc_time']
    GEO=result['Geometry']
    Pore=result['Porosity']
    
    rho = F.sum(axis=3)
    P=rho/3
    A=256*256
    viscosity=1/6
    OUT=np.mean(P[-1,:,:])
    IN=np.mean(P[-2,:,:])
    Q=np.mean(UX[-1,:,:])+np.mean(UY[-1,:,:])+np.mean(UZ[-1,:,:])
    K1=-Q*A*viscosity/(OUT-IN)
    OUT2=np.mean(P[2,:,:])
    IN2=np.mean(P[1,:,:])
    Q2=np.mean(UX[1,:,:])+np.mean(UY[1,:,:])+np.mean(UZ[1,:,:])
    K2=-Q2*A*viscosity/(OUT2-IN2)
    OUT3=np.mean(P[-1,:,:])
    IN3=np.mean(P[1,:,:])
    Q3out=np.mean(UX[-1,:,:])+np.mean(UY[-1,:,:])+np.mean(UZ[-1,:,:])
    Q3in=np.mean(UX[1,:,:])+np.mean(UY[1,:,:])+np.mean(UZ[1,:,:])
    Q3=Q3out-Q3in
    K3=-Q3*A*viscosity/(OUT3-IN3)
    print('Absolute permeability K1:',K1,'\n',
          'Initial permeability K2:',K2,'\n',
          'Effective permeability K3:',K3)
    
    f=open(save_path+'log.txt','a')
    f.write('Calc time: {:2.4f}'.format(Time))
    f.write('Porosity: {:2.4f}'.format(Pore))
    f.write('\n')
    f.write('Absolute permeability K1: {:2.4f} darcy'.format(K1))
    f.write('\n')
    f.write('Initial permeability K1: {:2.4f} darcy'.format(K2))
    f.write('\n')
    f.write('Effective permeability K1: {:2.4f} darcy'.format(K3))
    f.write('\n')
    f.close()
    
    #plot
    font = {'family' : 'Arial',
            'color'  : 'black',
            'weight' : 'normal',
            'size'   : 30,
            }
    #stream plot
    try:
        os.makedirs(save_path+'stream_plot')
    except OSError:
        pass
    stream_plot_path=save_path+'stream_plot/'
    plt.figure(figsize=(8,5),dpi=200)
    plt.imshow(~GEO[125,:,:],cmap=plt.cm.gray)
    color=2*(np.hypot(UX[125,:,:],UY[125,:,:]))
    x1,y1=meshgrid(arange(0,256),arange(0,256))
    streamplot(x1,y1,UX[125,:,:],UY[125,:,:],
               density=5,
               color=color,
               cmap=plt.cm.jet,
               arrowstyle='fancy',
               minlength=0.1,
               arrowsize=0.8,
               maxlength=10,
               linewidth=0.5)
    h=plt.colorbar(fraction=0.04,pad=0.05,shrink=0.9)
    h.ax.tick_params(labelsize=15)
    plt.axis('off')
    plt.savefig(stream_plot_path+'UX_UY_left.png',bbox_inches='tight')
    plt.figure(figsize=(8,5),dpi=200)
    plt.imshow(~GEO[:,125,:],cmap=plt.cm.gray)
    color=2*(np.hypot(UX[:,125,:],UZ[:,125,:]))
    streamplot(x1,y1,UX[:,125,:],UZ[:,125,:],
               density=2.5,
               color=color,
               cmap=plt.cm.jet,
               arrowstyle='fancy',
               minlength=0.1,
               arrowsize=0.8,
               maxlength=10,
               linewidth=0.5)
    h=plt.colorbar(fraction=0.04,pad=0.05,shrink=0.9)
    h.ax.tick_params(labelsize=15)
    plt.axis('off')
    plt.savefig(stream_plot_path+'UX_UZ_right.png',bbox_inches='tight')
    plt.figure(figsize=(8,5),dpi=200)
    plt.imshow(~GEO[:,:,125],cmap=plt.cm.gray)
    color=2*(np.hypot(UY[:,:,125],UZ[:,:,125]))
    streamplot(x1,y1,UY[:,:,125],UZ[:,:,125],
               density=3.5,
               color=color,
               cmap=plt.cm.jet,
               arrowstyle='fancy',
               minlength=0.1,
               arrowsize=0.8,
               maxlength=10,
               linewidth=0.5)
    h=plt.colorbar(fraction=0.04,pad=0.05,shrink=0.9)
    h.ax.tick_params(labelsize=15)
    plt.axis('off')
    plt.savefig(stream_plot_path+'UY_UZ_top.png',bbox_inches='tight')
    #UX plot
    try:
        os.makedirs(save_path+'UX_plot')
    except OSError:
        pass
    UX_plot_path=save_path+'UX_plot/'
    mins=[]
    mins.append(np.min(UX[125,:,:]))
    mins.append(np.min(UX[:,125,:]))
    mins.append(np.min(UX[:,:,125]))
    vmin=np.min(mins)
    maxs=[]
    maxs.append(np.max(UX[125,:,:]))
    maxs.append(np.max(UX[:,125,:]))
    maxs.append(np.max(UX[:,:,125]))
    vmax=np.max(maxs)
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.figure(figsize=(20,6),dpi=300)
    plt.subplot(1,3,1)
    plt.contourf(UX[125,:,:],cmap='jet',norm=norm)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.contourf(UX[:,125,:],cmap='jet',norm=norm)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.contourf(UX[:,:,125],cmap='jet',norm=norm)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    h=plt.colorbar(fraction=0.04,pad=0.05,shrink=0.9)
    h.ax.tick_params(labelsize=15)
    plt.savefig(UX_plot_path+'UX_left_right_top.png',bbox_inches='tight')
    
    #UY plot
    try:
        os.makedirs(save_path+'UY_plot')
    except OSError:
        pass
    UY_plot_path=save_path+'UY_plot/'
    mins=[]
    mins.append(np.min(UY[125,:,:]))
    mins.append(np.min(UY[:,125,:]))
    mins.append(np.min(UY[:,:,125]))
    vmin=np.min(mins)
    maxs=[]
    maxs.append(np.max(UY[125,:,:]))
    maxs.append(np.max(UY[:,125,:]))
    maxs.append(np.max(UY[:,:,125]))
    vmax=np.max(maxs)
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.figure(figsize=(15,4),dpi=300)
    plt.subplot(1,3,1)
    plt.contourf(UY[125,:,:],cmap='jet',norm=norm)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.contourf(UY[:,125,:],cmap='jet',norm=norm)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.contourf(UY[:,:,125],cmap='jet',norm=norm)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    h=plt.colorbar(fraction=0.04,pad=0.05,shrink=0.9)
    h.ax.tick_params(labelsize=15)
    plt.savefig(UY_plot_path+'UY_left_right_top.png',bbox_inches='tight')
    
    #UZ plot
    try:
        os.makedirs(save_path+'UZ_plot')
    except OSError:
        pass
    UZ_plot_path=save_path+'UZ_plot/'
    mins=[]
    mins.append(np.min(UZ[125,:,:]))
    mins.append(np.min(UZ[:,125,:]))
    mins.append(np.min(UZ[:,:,125]))
    vmin=np.min(mins)
    maxs=[]
    maxs.append(np.max(UZ[125,:,:]))
    maxs.append(np.max(UZ[:,125,:]))
    maxs.append(np.max(UZ[:,:,125]))
    vmax=np.max(maxs)
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.figure(figsize=(15,4),dpi=300)
    plt.subplot(1,3,1)
    plt.contourf(UZ[125,:,:],cmap='jet',norm=norm)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.contourf(UZ[:,125,:],cmap='jet',norm=norm)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.contourf(UZ[:,:,125],cmap='jet',norm=norm)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    h=plt.colorbar(fraction=0.04,pad=0.05,shrink=0.9)
    h.ax.tick_params(labelsize=15)
    plt.savefig(UZ_plot_path+'UZ_left_right_top.png',bbox_inches='tight')
    
    #U_SQU plot
    try:
        os.makedirs(save_path+'U_SQU_plot')
    except OSError:
        pass
    U_SQU_plot_path=save_path+'U_SQU_plot/'
    mins=[]
    mins.append(np.min(U_SQU[125,:,:]))
    mins.append(np.min(U_SQU[:,125,:]))
    mins.append(np.min(U_SQU[:,:,125]))
    vmin=np.min(mins)
    maxs=[]
    maxs.append(np.max(U_SQU[125,:,:]))
    maxs.append(np.max(U_SQU[:,125,:]))
    maxs.append(np.max(U_SQU[:,:,125]))
    vmax=np.max(maxs)
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.figure(figsize=(15,4),dpi=300)
    plt.subplot(1,3,1)
    plt.contourf(U_SQU[125,:,:],cmap='jet',norm=norm)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.contourf(U_SQU[:,125,:],cmap='jet',norm=norm)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.contourf(U_SQU[:,:,125],cmap='jet',norm=norm)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    h=plt.colorbar(fraction=0.04,pad=0.05,shrink=0.9)
    h.ax.tick_params(labelsize=15)
    plt.savefig(U_SQU_plot_path+'U_SQU_left_right_top.png',bbox_inches='tight')
    
    
'''
 8. LBM_3D_Analysis
LBM_3D_Analysis(path,out_folder)
'''
