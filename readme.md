1. Image_preprocess
2. Augmentation
3. Segmentation
4. LBM 2D,3D
5. Reconstruction
5.1 DCGAN
5.2 Dual-GAN
5.3 Disco-GAN
5.4 Cycle-GAN
6 cross-generation

Functions:
'''
 1. Segmentation training set prepare
path=train_set_create(file_path,out_folder)
label_me(if_labelme)
cmd->labelme_json_to_dataset
mask_path,Classes=json2dataset(json_path,label_number,out_folder)
 2. Segmentation model predict
model_predict_set_create(file_path,out_folder)
'''
'''
 3. Augmentation_dataset for few-shot learning
save_path_list=Augmentation_dataset(file_path,out_folder,json_path,label_number,train_num,test_num,val_num)
'''
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
'''
 7. LBM_2D_Analysis
LBM_2D_Analysis(path_all,out_folder)
'''
'''
 8. LBM_3D_Analysis
LBM_3D_Analysis(path,out_folder)
'''
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
'''
 12. Dual_GAN generate from promoted translation style
Dual_GAN(out_folder=out_folder,
         dataset_name=dataset_name,
         dataset_path=dataset_path,
         checkpoint_interval=checkpoint_interval,
         sample_interval=sample_interval,
         n_epochs=n_epochs,
         batch_size= batch_size,
         lr=learning_rate,
         img_size=generate_image_size,
         channels=channels,
         pre_trained=pre_trained,
         trained_epoch=trained_epoch)
'''
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
