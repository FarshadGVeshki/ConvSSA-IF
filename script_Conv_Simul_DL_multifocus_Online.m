% demo for online single dictionary learning based on convolutional sparse
% approximation (multifocus images as training data)

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
%% Parameters
lamb = 0.1; % L1 regularization parameter
sig= 5; % penalty parameter for OCDL (fixed)
%% history arrays
Hist = 0; inv_Mtx = 0;
%% Online Convolutional Dictionary Learning (OCDL) with Convolutional Simultaneous Sparse Approximation (CSSA)
tic
for n =1:10  
    S1 = imresize(single(rgb2gray(imread(['Data\LytroDataset\lytro-' num2str(n,'%02.0f') '-A.jpg'])))/255,1); % input 1
    S2 = imresize(single(rgb2gray(imread(['Data\LytroDataset\lytro-' num2str(n,'%02.0f') '-B.jpg'])))/255,1); % input 2
    [h, w] =size(S1);
    % lowpass filtering (removing the low frequency components)
    fltlmbd = 10;
    [~, S1] = lowpass(S1, fltlmbd); 
    [~, S2] = lowpass(S2, fltlmbd);    
    S = cat(3,S1,S2);       
    %% CSSA
    [X, res] = ConvSSA_SparseDense(D, S, lamb);
    %% OCDL
    for i =1:2               
        [Hist,inv_Mtx] = hist_update(Hist,inv_Mtx,X(:,:,:,i),S(:,:,i),sig,2*(n-1)+i); % Updating the histories
        [D,r] = OCDL(D,Hist,inv_Mtx,sig); % updating the filters
    end            
end
toc
% save(['Dictionaries\D_K' num2str(K) '_MF.mat'],'D')
%%
Im_D = dict2image(D,dcfilter);
figure(1)
imshow(Im_D)