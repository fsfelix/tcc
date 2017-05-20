function [logLike] = gloglike(x,mu,sigma);

% GLOGLIKE   2D Gaussian log-likelihood
%
%    GLOGLIKE(X,MU,SIGMA) returns the log-likelihood
%    of the point or set of points X with respect to
%    a Gaussian process with mean MU and covariance
%    SIGMA. MU is a 2D vector, SIGMA is a 2*2 matrix
%    and X is a N*2 matrix.
%
%    See also MEAN, COV, GAUSSPDF

N=size(x,1);
invSig = inv(sigma);
mu = mu(:)';

x = x-repmat(mu,N,1);

logLike = sum( sum( ((x*invSig).*x) ) );

logLike =  - 0.5 * ( logLike + N*log(det(sigma)) + 2*N*log(2*pi) );
