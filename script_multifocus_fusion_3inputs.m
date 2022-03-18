% script for multi-focus image fusion (three input images) using convolutional simaltaneous
% sparse coding
clear
clc
addpath('.\utility')
%% Parameters
lamb = 0.01; % L1 regularization parameter
%%
load('Dictionaries\D_K16_MF.mat')
%%
n=1;
S1_rgb = imread(['Data\LytroDataset\Triple Series\lytro-' num2str(n,'%02.0f') '-A.jpg']); % input 1
S2_rgb = imread(['Data\LytroDataset\Triple Series\lytro-' num2str(n,'%02.0f') '-B.jpg']); % input 2
S3_rgb = imread(['Data\LytroDataset\Triple Series\lytro-' num2str(n,'%02.0f') '-C.jpg']); % input 2
% fusing rgb layers separately
for c =1:3
S1 = single(S1_rgb(:,:,c))/255; % input 1
S2 = single(S2_rgb(:,:,c))/255; % input 2
S3 = single(S3_rgb(:,:,c))/255; % input 2
S = cat(3,S1,S2,S3);
F(:,:,c) = Multifocus_fusion_ConvSSA(S,D,lamb);
end
imwrite(F,'Results\fused_multifocus_triple_input.png');
%%
figure(1)
subplot 221
imshow(S1_rgb,[])
xlabel('input 1')
subplot 222
imshow(S2_rgb,[])
xlabel('input 2')
subplot 223
imshow(S3_rgb,[])
xlabel('input 3')
subplot 224
imshow(F,[])
xlabel('fused image')