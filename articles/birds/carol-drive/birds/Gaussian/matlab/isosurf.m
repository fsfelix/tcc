
% ISOSURF Plotting of gaussian level lines
%
%    This script plots level lines at iso-likelihood levels
%    for the models of class /i/ and class /e/.
%

[xx,yy] = meshgrid(0:10:2000, 1000:10:3000);

range = linspace(-30, -12, 10);

% First plot with full covariance
zz1 = reshape( gausspdf([xx(:) yy(:)],mu_i,sigma_i) , size(xx) );
zz2 = reshape( gausspdf([xx(:) yy(:)],mu_e,sigma_e) , size(xx) );
subplot(2,1,1);
[c,h] = contour3(xx,yy,log(zz1),range); hold on;
[c,h] = contour3(xx,yy,log(zz2),range);
colormap(jet);
title('Iso-likelihood lines with DIFFERENT covariances');
xlabel('F1')
ylabel('F2')

set(gca,'dataaspectratio',[1 1 1]);
view(2);

% Second plot with equal covariance, set to sigma_e
zz1 = reshape( gausspdf([xx(:) yy(:)],mu_i,sigma_e) , size(xx) );
subplot(2,1,2);
[c,h] = contour3(xx,yy,log(zz1),range); hold on;
[c,h] = contour3(xx,yy,log(zz2),range);
colormap(jet);
title('Iso-likelihood lines with EQUAL covariances');
xlabel('F1')
ylabel('F2')

set(gca,'dataaspectratio',[1 1 1]);
view(2);

% Printing settings
set(gcf,'paperunits','cent','papertype','a4','paperpos',[0 0 21 29.7]);