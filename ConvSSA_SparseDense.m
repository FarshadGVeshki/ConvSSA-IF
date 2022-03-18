function [X, res] = ConvSSA_SparseDense(D, S, lamb, opts)
% CONVOLUTIONAL SIMULTANEOUS SPARSE APPROXIMATION (with sparse-dense structure: nonzero entries are exactly the same) 
% (only L2,1-norm regularization is used)
%
%
% Inputs:
%   S:                T sets of N multimodal (multi measurement) input signals of size h*w (S is h x w x N, T)
%                     signal with modality i from the set t (s_i,t) is in s(:,:,i,t) 
%   D:                Convolutional dictionaries with size m x m x K x n. For single mode dictionaries n=1.
%   lamb:             sparsity regularization
%   (optionals:)
%   opts.MaxIter      maximum algorithm iterations (default 150)
%
% Outputs
%   X                 sparse codes with identical support (h x w x K x N, T)
%   res               iterations details

[h,w,N,T] = size(S);
S = reshape(S,[h,w,1,N,T]);
K = size(D,3);

if nargin < 4
    opts = [];
end

if ~isfield(opts,'MaxIter')
    opts.MaxIter = 200;
end

if ~isfield(opts,'rho')
    opts.rho = 1;
end
if ~isfield(opts,'AutoRho')
    opts.AutoRho = 1; % varying rho (ADMM extension)
end
if ~isfield(opts,'RhoUpdateCycle')
    opts.RhoUpdateCycle = 1;
end
if ~isfield(opts,'Xinit')
    opts.Xinit = zeros(h,w,K,N,T,'single');
end
if ~isfield(opts,'relaxParam')
    opts.relaxParam = 1.8; % over relaxion parameter (ADMM extension)
end
if ~isfield(opts,'eAbs')
    opts.eAbs = 1e-3;
end
if ~isfield(opts,'eRel')
    opts.eRel = 1e-3;
end

%% initialization
alpha = opts.relaxParam;
Sf = fft2(S);

X = opts.Xinit; % sparse code

rho = opts.rho;
RhoUpdateCycle = opts.RhoUpdateCycle;

epri = opts.eAbs;
edua = opts.eAbs;

MaxIter = opts.MaxIter;
r = inf; s = inf;
res.iterinf = [];
mu = 2; % varying rho parameter
tau = 1.2; % varying rho parameter

vec = @(x) x(:);
itr = 1;

%% CDL CYCLES
tsrt = tic;

U = X; % scaled dual variable
Df = fft2(D, h, w);

while itr<=MaxIter && (r > epri || s > edua)
    %%% ADMM iterations    
    Xprv = X;
    Y  = Y_update(fft2(X-U),Df,Sf,rho) ; % Y update: convolutionl LS regression
    Yr = alpha * Y + (1-alpha)*X; % relaxation
    X = prox_L12(Yr+U, lamb/rho); % X update: shrinkage
    U = Yr - X + U; % U update (lagrangian multipliers)
    titer = toc(tsrt);
    %%    
    %_________________________residuals CSC__________________________
    nY = norm(Y(:));nX = norm(X(:));nU = norm(U(:));
    r = norm(vec(Y-X))/(max(nX,nY)); % primal residulal
    s = norm(vec(Xprv-X))/nU; % dual residual    
    %_________________________rho update_____________________________
    if opts.AutoRho && rem(itr,RhoUpdateCycle)==0
        [rho,U] = rho_update(rho,r,s,mu,tau,U);
    end 
    %_________________________progress_______________________________
    rPow = sum(vec(abs(sum(Df.*fft2(X),3)-Sf).^2))/(h*w); % residual power
    L12 = sum(vec(sqrt(sum(X.^2,4))));   
    fval = rPow/2 + lamb*(L12); % functional value
    res.iterinf = [res.iterinf; [itr fval rPow L12  r s rho titer]];    
    itr = itr+1;
end
end

function Y = prox_L12(X, kappa) % proximal operator for L2,1-norm 
Y = zeros(size(X));
for i = 1:size(X,5)
    invNA = 1 - kappa./max(kappa,sqrt(sum(X(:,:,:,:,i).^2,4)));
    Y(:,:,:,:,i) = X(:,:,:,:,i).*invNA;
end
end

function Z  = Y_update(Wf,Df,Sf,rho)
C = conj(Df)./(sum(abs(Df).^2,3)+rho);
Rf = Sf - sum(Wf.*Df,3); % residual update
Zf = Wf + C.*Rf; % X update
Z  = ifft2(Zf,'symmetric');
end


function [rho,V] = rho_update(rho,r,s,mu,tau,V)
% varying penalty parameter
a = 1;
if r > mu*s
    a = tau;
end
if s > mu*r
    a = 1/tau;
end
rho_ = a*rho;
if rho_>1e-4
    rho = rho_;
    V = V/a;
end
end
