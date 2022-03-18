function F = Multifocus_fusion_ConvSSA(S_org,D,lamb)
% Multifocus image fusion based on convolutional simultaneous sparse approximation
%
% Inputs
% S: n stacked multifocus images of size h*w
% D: convolutional dictionary (m*m*K)
% lamb: sparse regularization parameter
%
% Output:
% F: fused all-in-focus image


[h,w,n] = size(S_org);
% lowpass filtering
Sb = zeros(h,w,n);
Sh = zeros(h,w,n);
fltlmbd = 5;
for i = 1:n
[Sbt, Sht] = lowpass(S_org(:,:,i), fltlmbd);
Sb(:,:,i) = Sbt;
Sh(:,:,i) = Sht;
end

%% CSSA
opts.eAbs = 1e-3;
opts.eRel = 1e-3;
[X, ~] = ConvSSA_SparseDense(D, Sh, lamb,opts);
%% Fusion
% low-freq fusion
g_size = 20;
g = ones(g_size)/g_size^2;
L1 = zeros(h,w,n);
for i=1:n
L1(:,:,i) = sum(abs(X(:,:,:,i)),3); L1=imfilter(L1,g);
end
L1max =max(L1,[],3); 
ind0 = (L1max == 0);
L1max = double(L1max);
L1max(ind0) = -1;
M = double(L1 == L1max);
M(repmat(ind0,1,1,n)) = 1/n;
Fb = sum(M.*Sb,3);
% high-freq fusion
Xf = sum(X.*double(abs(X) == max(abs(X),[],4)),4);
Fh = ifft2(sum(fft2(D,h,w).*fft2(Xf),3),'symmetric');
% fused image
F = uint8((Fb + Fh)*255);