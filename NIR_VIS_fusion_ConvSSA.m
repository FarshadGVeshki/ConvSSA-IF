function F = NIR_VIS_fusion_ConvSSA(Svis,Snir,D,gamma1,gamma2)
% NIR-RGB image fusion based on convolutional simultaneous sparse approximation
%
% Inputs
% Svis: RGB image of size h*w*3
% Snir: NIR image of size h*w*3 (greyscale image)
% D: convolutional dictionary (m*m*K)
% gamma1: element sparsity regularization parameter
% gamma2: row sparsity regularization parameter
%
% Output:
% F: enhanced RGB image

h = size(Svis,1);
w = size(Svis,2);

Svis_ycbcr = rgb2ycbcr(single(Svis)/255);
Svis_grey = Svis_ycbcr(:,:,1);
Snir = single(rgb2gray(Snir))/255;

% lowpass filtering
fltlmbd = 10;
[Fb, Svis_h] = lowpass(Svis_grey, fltlmbd);
[~, Snir_h] = lowpass(Snir, fltlmbd);

S = cat(3,Svis_h,Snir_h);
%% CSSA
opts.eAbs = 1e-3;
opts.eRel = 1e-3;
[X, ~] = ConvSSA_SparseSparse(D, S, gamma1,gamma2,opts);
%% Fusion
Xvis = X(:,:,:,1);
Xnir = X(:,:,:,2);
Xf_vis = Xvis;
Xf_nir = zeros(size(Xnir));
Xf_vis(abs(Xnir)>abs(Xvis)) = 0;
Xf_nir(abs(Xnir)>abs(Xvis)) = Xnir(abs(Xnir)>abs(Xvis));

Fh_y = ifft2(sum(fft2(D(:,:,:,1),h,w).*fft2(Xf_vis),3),'symmetric') + ifft2(sum(fft2(D(:,:,:,2),h,w).*fft2(Xf_nir),3),'symmetric');
Fh = ycbcr2rgb(cat(3,Fh_y+Fb,Svis_ycbcr(:,:,2:3)));
% fused image
F = uint8(Fh*255);
end