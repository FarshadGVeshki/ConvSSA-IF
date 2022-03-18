function [D,res] = OCDL(D,H,invM,sig,opts)
%%% Online Convolutional Dictionary Learning
% reference:
% Y. Wang, Q. Yao, J. T. Kwok and L. M. Ni, "Scalable Online Convolutional Sparse Coding," 
% in IEEE Transactions on Image Processing, vol. 27, no. 10, pp. 4850-4859, Oct. 2018, doi: 10.1109/TIP.2018.2842152.
%
% Pramameters:
% D: Dictionry
% H: history array 3D
% invM: 4D array of inversed matrices
% sig: penalty parameter (ADMM)

if nargin < 5
    opts = [];
end
if ~isfield(opts,'MaxIter')
    opts.MaxIter = 200;
end
if ~isfield(opts,'dcfilter')
    opts.dcfilter = 0;
end
if ~isfield(opts,'relaxParam')
    opts.relaxParam = 1.8;
end
if ~isfield(opts,'eAbs')
    opts.eAbs = 1e-3;
end
if ~isfield(opts,'eRel')
    opts.eRel = 1e-3;
end

MaxIter = opts.MaxIter; % max iterations
epri = opts.eAbs;
edua = opts.eRel;
alpha = opts.relaxParam;
dcfilter = opts.dcfilter;

[m,~,K] = size(D);
h = size(H,1); w = size(H,2);

D = padarray(D,[h-m w-m],'post');
U = zeros(size(D)); % scaled lagrangians
vec = @(x) x(:);
itr = 1; r = inf; s = inf; res = [];
tstrt = tic;
while itr < MaxIter && (r > epri || s > edua)
    %% ADMM steps 
    Dprv = D;
    G = ifft2(sum(invM.*reshape(H+sig*sqrt(w*h)*fft2(D-U),[h w 1 K]),4) ,'symmetric');
    Gr = alpha*G+(1-alpha)*D; % relaxation
    D = D_proj(Gr+U,m,h,w);
    if dcfilter,D(:,:,1)=Dprv(:,:,1); end
    U = Gr-D+U;     
    %% Progress
    titer = toc(tstrt);
    r = norm(vec(G-D)); % primal residulal
    s = norm(vec(Dprv-D)); % dual residulal
    res = [res;[itr r s titer]];
    itr = itr+1;
end
D = D(1:m,1:m,:); % removing the zeropadding
D = D./sqrt(sum(D.^2,1:2)); % normalzing the filters
end

function D = D_proj(D,m,H,W) % projection on the unit L2 ball
D = padarray(D(1:m,1:m,:,:),[H-m W-m],'post');
D  = D./max(sqrt(sum(D.^2,1:2)),1);
end
