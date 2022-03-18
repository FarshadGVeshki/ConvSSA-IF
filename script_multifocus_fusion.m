% script for multi-focus image fusion (two input images) using convolutional simaltaneous
% sparse coding
clear
clc
addpath('.\utility')
%% Parameters
lamb = 0.01; % L1 regularization parameter
%%
load('Dictionaries\D_K16_MF.mat')
%%
n=11;
S1_rgb = imread(['Data\LytroDataset\lytro-' num2str(n,'%02.0f') '-A.jpg']); % input 1
S2_rgb = imread(['Data\LytroDataset\lytro-' num2str(n,'%02.0f') '-B.jpg']); % input 2
% fusing rgb layers separately
for c =1:3
S1 = single(S1_rgb(:,:,c))/255; % input 1
S2 = single(S2_rgb(:,:,c))/255; % input 2
S = cat(3,S1,S2);
F(:,:,c) = Multifocus_fusion_ConvSSA(S,D,lamb);
end
%%
imwrite(F,'Results\fused_multifocus.png');
%%
figure(1)
subplot 131
imshow(S1_rgb,[])
xlabel('input 1')
subplot 132
imshow(S2_rgb,[])
xlabel('input 2')
subplot 133
imshow(F,[])
xlabel('fused image')