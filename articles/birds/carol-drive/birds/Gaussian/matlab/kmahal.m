function [] = kmahal(fun,varargin)

% KMAHAL K-means algorithm exploration tool
%
%   Launch it with KMAHAL(DATA,NCLUST) where DATA is the matrix
%   of observations (one observation per row) and NCLUST is the
%   desired number of clusters.
%
%   The clusters are initialized with a heuristic that spreads
%   them randomly around mean(DATA) with standard deviation
%   sqrtm(cov(DATA)).
%
%   If you want to set your own initial clusters, use
%   KMAHAL(DATA,MEANS) where MEANS is a cell array containing
%   NCLUST initial mean vectors.
%
%   Example: for two clusters
%     means{1} = [1 2]; means{2} = [3 4];
%     kmahal(data,means);
%


global data mmeans vvars ppriors piConst numClust numPts whereMin whereBefore circle hc hp hb hl cmap;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isstr(fun),
  
  data = fun;
  [numPts,dim] = size(data);
  
  startMean = mean(data);
  startVvars = cov(data);
  startDev = sqrtm(startVvars);

  % Init clusters and likelihood
  if iscell(varargin{1}),
    
    numClust = length(varargin{1});
    mmeans = varargin{1};
    if (length(varargin)<2),
      for i=1:numClust,
	vvars{i} = startVvars;
      end;
    else,
      vvars = varargin{2};
    end;

    if (length(varargin)<3),
      ppriors(1:numClust) = 1 / numClust;
    else,
      ppriors = varargin{3};      
    end;    
    
  else,
    
    numClust = varargin{1};

    for i=1:numClust,
      mmeans{i} = randn(1,dim) * startDev + startMean;
      vvars{i} = startVvars;
      ppriors(i) = 1 / numClust;
    end;
  
  end;
  
  % Make colormap for classification
  cmap = hsv(numClust);
  
  piConst = -0.5 * numPts * log( (2*pi)^dim );
  % Compute distance
  for i=1:numClust,
    x = data - repmat(mmeans{i},numPts,1);
    invSig = inv(vvars{i});
    %dist(i,:) = sum( ((x * invSig) .* x)' ) + log(det(vvars{i}));
    dist(i,:) = sum( ((x * invSig) .* x)' );
  end;
  % Find minimum distance
  [minDist,whereMin] = min(dist);
  whereBefore = zeros(size(whereMin));
  
  
  % Init figure
  hf = figure('name','Kmahal Explorer','numbertitle','off');
  
  % Attribution plot
  subplot('position',[ 0 0.05 0.45 0.9 ]);
  %plot(data(:,1),data(:,2),'y+');
  set(gca,'xlim',[0 1000],'ylim',[0 3000],'dataaspectratio',[1 1 1], ...
      'drawmode','fast');
  xlabel('F1 (Hz)'); ylabel('F2 (Hz)');
  title('Attribution of points to clusters');
  grid on; zoom on; hold on;
  
  circle = [cos(linspace(-pi, pi, 100)') sin(linspace(-pi, pi, 100)')];
  for i=1:numClust,
    subData = data(whereMin==i,:);
    hp(i) = line( subData(:,1), subData(:,2), ...
	'linestyle','none','marker','+','color',cmap(i,:));

    hc(i) = line(mmeans{i}(1),mmeans{i}(2), 10, ...
	'marker','o','markersize',12,'color',[1 1 1],'linew',2);
  end;
  
  
  % Likelihood plot
  subplot('position',[ 0.51 0.5 0.45 0.45 ]);
  hl = plot(0,NaN,'yo-');
  grid on;
  set(gca, 'drawmode','fast');
  zoom on;
  set(hl,'markersize',5);
  xlabel('Number of iterations');
  title('Cumulated distance J');

  
  % Init buttons
  btnWdth = 0.15;
  btnHt = 0.1;
  
  hb(1) = uicontrol('style','push', ...
      'units','norm', ...
      'pos', [0.5 0.3 btnWdth btnHt], ...
      'background', [0.8 0.8 0.8], ...
      'foreground', [0 0 0], ...
      'string','Iterate once', ...
      'callback', 'kmahal(''iterate'',1);', ...
      'visible', 'on');
  
  hb(2) = uicontrol('style','push', ...
      'units','norm', ...
      'pos', [0.65 0.3 btnWdth btnHt], ...
      'background', [0.8 0.8 0.8], ...
      'foreground', [0 0 0], ...
      'string','Iterate 5 times', ...
      'callback', 'kmahal(''iterate'',5);', ...
      'visible', 'on');
  
  hb(3) = uicontrol('style','push', ...
      'units','norm', ...
      'pos', [0.8 0.3 btnWdth btnHt], ...
      'background', [0.8 0.8 0.8], ...
      'foreground', [0 0 0], ...
      'string','Iterate 10 times', ...
      'callback', 'kmahal(''iterate'',10);', ...
      'visible', 'on');
  
  hb(4) = uicontrol( 'style','text', ...
      'units','norm', 'position',[0.5 0.15 0.45 0.15], ...
      'string','Push a button to start iterating...', ...
      'foreground', [1 0.3 0.3], 'background',[0 0 0], ...
      'fontsize',16,'fontweight','bold' );
      
  hb(5) = uicontrol('style','push', ...
      'units','norm', ...
      'pos', [0.5 0.05 btnWdth btnHt], ...
      'background', [0.8 0.8 0.8], ...
      'foreground', [0 0 0], ...
      'string','Iterate until conv.', ...
      'callback', 'kmahal(''iterate'',10000);', ...
      'visible', 'on');
  
  hb(6) = uicontrol('style','push', ...
      'units','norm', ...
      'pos', [0.8 0.05 btnWdth btnHt], ...
      'background', [0.8 0.8 0.8], ...
      'foreground', [0 0 0], ...
      'string','Close', ...
      'callback', 'kmahal(''stop'');', ...
      'visible', 'on');
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(fun,'iterate'),
  
  nIter = varargin{1};
  
  set(hb,'enable','off');
  
  for k=1:nIter,
    
    % Check for convergence
    if all(all(whereBefore == whereMin)),

      % Send message
      xd = get(hl,'xdata'); yd = get(hl,'ydata');
      str = sprintf('Convergence has been reached in %i iterations.\n\nTotal likelihood at the end is %1.2e.', xd(end), yd(end) );
      set(hb(4),'string',str);

      % Update plots for likelihood
%      for i=1:numClust,
%	% Likelihood
%	subData = data(whereMaxLike==i,:);
%	set( hp(i), 'xdata', subData(:,1), 'ydata',subData(:,2) );
%	
%	subplot('position',[ 0 0.05 0.45 0.9 ]);
%	ellip = circle * sqrtm(vvars{i}) + repmat(mmeans{i},100,1);
%	line(ellip(:,1),ellip(:,2),10*ones(size(ellip,1),1), ...
%	    'color',[1 1 1],'linew',2);
%	line(mmeans{i}(1),mmeans{i}(2), 10, ...
%	    'marker','+','markersize',10,'color',[1 1 1],'linew',2);
%	title('Maximum likelihood classes based on k-means clusters');
%      end;
      
      drawnow;
      
      % Send results to base workspace
      assignin('base','kmahal_result_means',mmeans);
      assignin('base','kmahal_result_vars',vvars);
      assignin('base','kmahal_result_priors',ppriors);
      disp(' ');
      disp([ 'KMAHAL: resulting means, variances and priors are now stored in' ...
	    ' the workspace variables kmahal_result_means,' ...
	    ' kmahal_result_vars and kmahal_result_priors.' ]);
      
      break;
      
    else,
      
      % Update clusters and plots for distance
      for i=1:numClust,
	% Distance
	subData = data(whereMin==i,:);
	set( hp(i), 'xdata', subData(:,1), 'ydata',subData(:,2) );
	
	% Update clusters
	mmeans{i} = mean( subData );
	vvars{i} = cov( subData );
	ppriors(i) = size(subData,1) / numPts;
	
	set( hc(i), 'xdata', mmeans{i}(1), 'ydata', mmeans{i}(2) );
      
      end;
      
      whereBefore = whereMin;
      for i=1:numClust,
	% Compute distance
	x = data - repmat(mmeans{i},numPts,1);
	invSig = inv(vvars{i});
	%dist(i,:) = sum( ((x * invSig) .* x)' ) + log(det(vvars{i}));
	dist(i,:) = sum( ((x * invSig) .* x)' );
	
%	% Compute log-likelihood
%	invSig = inv(vvars{i});
%	logLike(i,:) = -0.5 * ( sum( ((x*invSig).*x)' ) );
%	logLike(i,:) =  logLike(i,:) - 0.5 * log(det(vvars{i})) + log(ppriors(i));
      end;
      % Find minimum distance
      [minDist,whereMin] = min(dist);
%      % Find max likelihood
%      [maxLike,whereMaxLike] = max(logLike);
      
      % Update likelihood plot
      xd = get(hl,'xdata'); xd = [xd  xd(end)+1];
      %yd = get(hl,'ydata'); yd = [yd (sum(maxLike) + piConst)];
      yd = get(hl,'ydata'); yd = [yd sum(minDist)];
      set(hl,'xdata',xd,'ydata',yd);
	
      drawnow;
      
    end;
  
  end;
  
  if all(all(whereBefore == whereMin)),
    set(hb([4 6]),'enable','on');
  else,
    set(hb,'enable','on');
  end;
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(fun,'stop'),
  
  clear data mmeans vvars ppriors piConst numClust numPts whereMin whereBefore circle hc hp hb hl cmap;
  close(gcf);
  
else,

  error('Unknown function.');
  
end;
