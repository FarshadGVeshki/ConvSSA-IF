function [H,invMtx] = hist_update(H_old,invMtx_old,x,s,sig,t)
%%% Online Convolutional Dictionary Learning
% Updating the history matrices
% reference:
% Y. Wang, Q. Yao, J. T. Kwok and L. M. Ni, "Scalable Online Convolutional Sparse Coding," 
% in IEEE Transactions on Image Processing, vol. 27, no. 10, pp. 4850-4859, Oct. 2018, doi: 10.1109/TIP.2018.2842152.

[h,w,K] = size(x);
if isempty(H_old)
    invMtx_old = zeros(h,w,K,K);
    H_old = zeros(h,w,K);
end
xf = fft2(x);
P = h*w;
H = (1-1/t)*H_old + (1/t)*conj(xf).*fft2(s);
if t == 1
   invMtx = (repmat(reshape(eye(K),[1 1 K K]),[h w,1,1])-(((xf).*reshape(conj(xf),[h w 1 K]))./(sig*P+sum(conj(xf).*xf,3))))/(sig*P);
else
   invMtx = (t/(t-1))*(invMtx_old-(sum(invMtx_old.*reshape(xf,[h,w,1,K]),4).*sum(conj(xf).*invMtx_old,3))./(t-1+ reshape(sum(sum(conj(xf).*invMtx_old,3).*reshape(xf,[h,w,1,K]),4),[h w])));
end
end
