% demo code for using CSSA with only L1 and both L1 and L2,1 regularizations 

clear
clc
close all
rng('default')
addpath('.\utility')

%%
load('Dictionaries\D_K32_MF.mat')
NV = imread('Data\NIR_VIS\country_0000.jpg');
S1 = single(NV(204:585,11:586))/255; % input 1 (Visible light image)
S2 = single(NV(204:585,597:1172))/255; % input 2 (Near infrared image)
[h, w] =size(S1);
fltlmbd = 5;
[~, S1] = lowpass(S1, fltlmbd);
[~, S2] = lowpass(S2, fltlmbd);
S = cat(3,S1,S2);

%% L1,2 regularizaion (row-sparse)
lamb = 0.01;
[X, ~] = ConvSSA_SparseDense(D, S, lamb);

X1 = X(:,:,:,1);
X2 = X(:,:,:,2);

residPow = sum(abs(sum(fft2(D,h,w).*fft2(X),3)-fft2(reshape(S,[h,w,1,2]))).^2,'all')/(h*w);
Sparsity = nnz(X)/numel(X);
CommSupp = nnz(X1.*X2)/nnz(abs(X1)+abs(X2));

RES__CSSASD = [Sparsity CommSupp residPow];

%% L1 and L1,2 regularizaion (row-sparse and element sparse)
load('Dictionaries\D_K32_MM.mat')
gamma_1 = 0.001;
gamma_2 = 0.01;

[X, ~] = ConvSSA_SparseSparse(D, S, gamma_1, gamma_2);

X1 = X(:,:,:,1);
X2 = X(:,:,:,2);

residPow = sum(abs(sum(fft2(D,h,w).*fft2(X),3)-fft2(reshape(S,[h,w,1,2]))).^2,'all')/(h*w);
Sparsity = nnz(X)/numel(X);
CommSupp = nnz(X1.*X2)/nnz(abs(X1)+abs(X2));

RES_CSSASS = [gamma_1 gamma_2 Sparsity CommSupp residPow];


