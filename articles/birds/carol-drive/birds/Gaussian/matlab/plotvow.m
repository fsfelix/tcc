
% PLOTVOW Script for plotting the simulated vowels database.
%

hf = figure;
plot(a(:,1),a(:,2),'y+');
hold on;
set(gca,'xlim',[0 3000],'ylim',[0 3000],'dataaspectratio',[1 1 1e-6]);
xlabel('F1'); ylabel('F2'); grid on;
plot(e(:,1),e(:,2),'r+');
plot(i(:,1),i(:,2),'g+');
plot(o(:,1),o(:,2),'b+');
plot(y(:,1),y(:,2),'m+');
legend('/a/','/e/','/i/','/o/','/y/');
title('Simulated F1-F2 data for various vowels');
