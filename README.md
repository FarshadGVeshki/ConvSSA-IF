Convolutional Simultaneous Sparse Approximation with Applications to RGB-NIR Image Fusion

%

Simultaneous sparse approximation (SSA) seeks to represent a set of dependent signals using sparse vectors with identical supports. The SSA model has been used in various signal and image processing applications involving multiple correlated input signals. In this paper, we propose algorithms for convolutional SSA (CSSA) based on the alternating direction method of multipliers. Specifically, we address the CSSA problem with different sparsity structures and the convolutional feature learning problem in multimodal data/signals based on the SSA model. We evaluate the proposed algorithms by applying them to multimodal and multifocus image fusion problems.

%

Run the script files:

1 - script_ConvSSA_comparison.m: comparing Convolutional SSA (CSAA) with different sparsity structures
 
2 - script_Conv_Simul_DL_multifocus_Online.m : Online convolutional dictionary learning (CDL) based on CSSA for multifocus image fusion

3 - script_Conv_Simul_DL_multimodal_Online.m : Online convolutional dictionary learning (CDL) based on CSSA for NIR-RGB image fusion 

4 - script_NIR_VIS_fusion.m : NIR-RGB image fusion based on CSSA

5 - script_multifocus_fusion.m: Multifocus image fusion based on CSSA

6 - script_multifocus_fusion_3inputs.m: Multifocus image fusion with 3 inputs based on CSSA

%

The codes include:

The covolutional simultaneous sparse approxiamtion algorithm with sparse-dense structure: ConvSSA_SparseDense.m

The covolutional simultaneous sparse approxiamtion algorithm with sparse-sparse structure: ConvSSA_SparseSparse.m

The multifocus image fusion algorithm: Multifocus_fusion_ConvSSA.m

The NIR-RGB image fusion algorithm: NIR_VIS_fusion_ConvSSA.m

The online convolutional dictionary learning algorithm of Wang et. al. 2018: OCDL.m and hist_update.m

code for generating Gaussian random multiscale dictionaries: initdict.m

code for visualizing multiscale filters: dict2image.m

pre-learned dictionaries (.mat files in Dictionaries folder)

%

The code for lowpass filtering (lowpass.m) is taken from SPORCO toolbox.

%

Multifocus images are taken from the Lytro dataset

NIR-RGB images are taken from EPFL RGB-NIR scene dataset

%

Reference : FG Veshki, SA Vorobyov, Convolutional Simultaneous Sparse Approximation with Applications to RGB-NIR Image Fusion, 	arXiv:2203.09913, March 2022.


