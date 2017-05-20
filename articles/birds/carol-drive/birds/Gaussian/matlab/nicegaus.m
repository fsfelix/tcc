function [c,h] = plotgaus( mu, sigma, xrange, yrange, colspec );

if nargin < 3; xrange = [0 4000]; end;
if nargin < 4; yrange = [0 4000]; end;
if nargin < 5; colspec = [0 1 1]; end;
npts = 100;

stdev = sqrtm(sigma);

x = linspace( xrange(1), xrange(2), npts);
y = linspace( xrange(1), xrange(2), npts);

[X,Y] = meshgrid(x,y);

Z = gausspdf([X(:) Y(:)],mu,sigma);
Z = reshape(Z,size(X));

%axstate = get(gca,'nextplot');
%
%[h] = mesh(X,Y,Z,'facecolor','none');
%colormap(jet);
%
%set(gca,'nextplot','add');

[c,h] = contour3(X,Y,Z);
set(h,'linewidth',2,'edgecolor',colspec);

%set(gca,'nextplot',axstate);
%
%dasp = get(gca,'dataaspectratio');
%xl = get(gca,'xlim');
%zl = get(gca,'zlim');
%scal = max( [(zl(1) - zl(2)) / (xl(1) - xl(2))  dasp(3)] );
%set(gca,'dataaspectratio',[1 1 scal]);
