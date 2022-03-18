% script for near-infrared-visible-light image fusion image fusion using convolutional simaltaneous
% sparse coding
clear
clc
addpath('.\utility')
%% Parameters
gamma1 = 0.001; % L1 regularization parameter
gamma2 = 0.001; % L12 regularization parameter
%%
load('Dictionaries\D_K32_MM.mat')
%%
n=0;
NV = imread(['Data\NIR_VIS\country_00' num2str(n,'%02.0f') '.jpg']);

Svis = NV(204:585,11:586,:); % input 1 (Visible light image)
Snir = NV(204:585,597:1172,:); % input 2 (Near infrared image)

% fusion
F = NIR_VIS_fusion_ConvSSA(Svis,Snir,D,gamma1,gamma2);
imwrite(F,'Results\multimodal_fused.png');
imwrite(Svis,'Results\multimodal_visible.png');
imwrite(Snir,'Results\multimodal_near_infrared.png');

%%
figure(1)
subplot 131
imshow(Svis,[])
xlabel('visible image')
subplot 132
imshow(Snir,[])
xlabel('NIR image')
subplot 133
imshow(F,[])
xlabel('fused image')