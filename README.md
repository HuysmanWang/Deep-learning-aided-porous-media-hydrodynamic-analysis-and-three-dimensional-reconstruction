# Deep-learning-aided-porous-media-hydrodynamic-analysis-and-three-dimensional-reconstruction
The study of hydrodynamic behavior and water-rock interaction mechanisms is typically characterized by high computational efficiency requirements, to allow for the fast and accurate extraction of structural information. Therefore, we chose to use deep learning models to achieve these requirements. In this paper we started by comparing the image segmentation performance of a series of autoencoder architectures on complex geometries of porous media. The goal was to extract hydrodynamic connectivity channels and the mineral composition of rock samples on SEM (Scanning electron microscopy) data, obtained with a 0.97 accuracy. We then focused on improving the computational efficiency of LBM by using GPU acceleration, which allowed us to rapidly simulate structural flow field features of complex porous media. The results obtained showed that we were able to improve the computational efficiency by a factor of 30 in our device environment. We subsequently employed a SWD-Cycle-GAN technique to migrate sedimentation features to the initial 2D structure slices to reconstruct a 3D (three-dimensional) porous media geometry, that fits the depositional features more closely. Overall, we propose a new method for 3D structure reconstruction and permeability performance analysis of porous media, based on deep learning. The proposed method is fast, efficient and accurate.
