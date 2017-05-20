function [z] = gausspdf(x,mu,sigma);

% GAUSSPDF  Values of the Gaussian probability density
%           function
%
%    GAUSSPDF(X,MU,SIGMA) returns the likelihood
%    of the point or set of points X with respect to
%    a Gaussian process with mean MU and covariance
%    SIGMA. MU is a 1*D vector (where D is the dimension
%    of the process), SIGMA is a D*D matrix
%    and X is a N*D matrix where each row is a
%    D-dimensional point.
%
%    See also MEAN, COV, GLOGLIKE

[N,D] = size(x);
if (min(N,D) == 1), x=x(:)'; end;

invSig = inv(sigma);
mu = mu(:)';

x = x-repmat(mu,N,1);

z = sum( ((x*invSig).*x), 2 );

z = exp(-0.5*z) / sqrt((2*pi)^D * det(sigma));
