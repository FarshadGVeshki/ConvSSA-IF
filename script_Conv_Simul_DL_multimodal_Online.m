% demo for online multimodal dictionary learning based on convolutional sparse
% approximation (NIR-RGB images as training data)

clear
clc
close all
rng('default')
addpath('.\utility')
%% Dictionary size
K = 16; % number of filters
Q = 8; % filter size
dcfilter = 0;
D = initdict(Q,K,dcfilter); % Initializing the dictionary with Gaussian random filters
D = repmat(D,[1 1 1 2]);
%% Parameters
gamma1 = 0.001; % L1 regularization parameter
gamma2 = 0.01; % L2,1 regularization parameter
sig= 5; % penalty parameter for OCDL (fixed)
%% history arrays
Hist1 = 0; inv_Mtx1 = 0;
Hist2 = 0; inv_Mtx2 = 0;
%% Online Convolutional Dictionary Learning (OCDL) with Convolutional Simultaneous Sparse Approximation (CSSA)
tic

for n =1:10
    NV = imread(['Data\NIR_VIS\indoor_00' num2str(n,'%02.0f') '.jpg']);
    S1 = imresize(single(NV(204:585,11:586))/255,1); % input 1 (Visible light image) images can be resized to reduce the memory usage
    S2 = imresize(single(NV(204:585,597:1172))/255,1); % input 2 (Near infrared image)
    
    [h, w] =size(S1);
    % lowpass filtering (removing the low frequency components)
    fltlmbd = 10;
    [~, S1] = lowpass(S1, fltlmbd); 
    [~, S2] = lowpass(S2, fltlmbd);    
    S = cat(3,S1,S2);       
    %% CSSA
    [X, res] = ConvSSA_SparseSparse(D, S, gamma1, gamma2);
    %% OCDL              
    [Hist1,inv_Mtx1] = hist_update(Hist1,inv_Mtx1,X(:,:,:,1),S1,sig,n); % Updating the histories
    [D_temp,r1] = OCDL(D(:,:,:,1),Hist1,inv_Mtx1,sig); % updating the filters
    D(:,:,:,1) = D_temp;
    [Hist2,inv_Mtx2] = hist_update(Hist2,inv_Mtx2,X(:,:,:,2),S2,sig,n); % Updating the histories
    [D_temp,r2] = OCDL(D(:,:,:,2),Hist2,inv_Mtx2,sig); % updating the filters
    D(:,:,:,2) = D_temp;
end
toc
% save(['Dictionaries\D_K' num2str(K) '_MM_SS.mat'],'D')
%%
Im_D1 = dict2image(D(:,:,:,1),dcfilter);
Im_D2 = dict2image(D(:,:,:,2),dcfilter);
figure(1)
subplot 121
imshow(Im_D1)
subplot 122
imshow(Im_D2)