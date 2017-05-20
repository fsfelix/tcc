function [] = viterb(fun,varargin)

% VITERB Viterbi version of the EM algorithm
%
%   Launch it with VITERB(DATA,NCLUST) where DATA is the matrix
%   of observations (one observation per row) and NCLUST is the
%   desired number of clusters.
%
%   The clusters are initialized with a heuristic that spreads
%   them randomly around mean(DATA) with standard deviation
%   sqrtm(cov(DATA)). Their initial covariance is set to cov(DATA).
%
%   If you want to set your own initial clusters, use
%   VITERB(DATA,MEANS,VARS) where MEANS and VARS are cell arrays
%   containing respectively NCLUST initial mean vectors and NCLUST
%   initial covariance matrices. In this case, the initial a-priori
%   probabilities are set equal to 1/NCLUST.
%
%   To set your own initial priors, use VITERB(DATA,MEANS,VARS,PRIORS)
%   where PRIORS is a vector containing NCLUST a priori probabilities.
%
%   Example: for two clusters
%     means{1} = [1 2]; means{2} = [3 4];
%     vars{1} = [2 0;0 2]; vars{2} = [1 0;0 1];
%     viterb(data,means,vars);
%


global data mmeans vvars ppriors piConst numClust numPts whereMax whereBefore circle hc hp hb hl cmap;

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
  for i=1:numClust,
    % Compute initial log-likelihood
    x = data - repmat(mmeans{i},numPts,1);
    invSig = inv(vvars{i});
    logLike(i,:) = -0.5 * ( sum( ((x*invSig).*x)' ) );
    logLike(i,:) =  logLike(i,:) - 0.5 * log(det(vvars{i})) + log(ppriors(i));
  end;
  % Find max likelihood
  [maxLike,whereMax] = max(logLike);

  % Init figure
  hf = figure('name','Viterbi-EM Explorer','numbertitle','off');
  
  % Attribution plot
  subplot('position',[ 0 0.05 0.45 0.9 ]);
  set(gca,'xlim',[0 1000],'ylim',[0 3000],'dataaspectratio',[1 1 1], ...
      'drawmode','fast');
  xlabel('F1 (Hz)'); ylabel('F2 (Hz)');
  title('Attribution of points to clusters');
  grid on; zoom on; hold on;
  
  circle = [cos(linspace(-pi, pi, 100)') sin(linspace(-pi, pi, 100)')];
  for i=1:numClust,
    subData = data(whereMax==i,:);
    hp(i) = line( subData(:,1), subData(:,2), ...
	'linestyle','none','marker','+','color',cmap(i,:));

    ellip = circle * sqrtm(vvars{i}) + repmat(mmeans{i},100,1);
    hc(i,1) = line(ellip(:,1),ellip(:,2),10*ones(size(ellip,1),1), ...
	'color',[1 1 1],'linew',2);
    hc(i,2) = line(mmeans{i}(1),mmeans{i}(2), 10, ...
	'marker','+','markersize',10,'color',[1 1 1],'linew',2);
  end;
  whereBefore = zeros(size(whereMax));
  
  
  % Likelihood plot
  subplot('position',[ 0.51 0.5 0.45 0.45 ]);
  hl = plot(0,NaN,'yo-');
  grid on;
  set(gca,'drawmode','fast');
  zoom on;
  set(hl,'markersize',5);
  xlabel('Number of iterations');
  title('Total Log-Likelihood');

  
  % Init buttons
  btnWdth = 0.15;
  btnHt = 0.1;
  
  hb(1) = uicontrol('style','push', ...
      'units','norm', ...
      'pos', [0.5 0.3 btnWdth btnHt], ...
      'background', [0.8 0.8 0.8], ...
      'foreground', [0 0 0], ...
      'string','Iterate once', ...
      'callback', 'viterb(''iterate'',1);', ...
      'visible', 'on');
  
  hb(2) = uicontrol('style','push', ...
      'units','norm', ...
      'pos', [0.65 0.3 btnWdth btnHt], ...
      'background', [0.8 0.8 0.8], ...
      'foreground', [0 0 0], ...
      'string','Iterate 5 times', ...
      'callback', 'viterb(''iterate'',5);', ...
      'visible', 'on');
  
  hb(3) = uicontrol('style','push', ...
      'units','norm', ...
      'pos', [0.8 0.3 btnWdth btnHt], ...
      'background', [0.8 0.8 0.8], ...
      'foreground', [0 0 0], ...
      'string','Iterate 10 times', ...
      'callback', 'viterb(''iterate'',10);', ...
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
      'callback', 'viterb(''iterate'',10000);', ...
      'visible', 'on');
  
  hb(6) = uicontrol('style','push', ...
      'units','norm', ...
      'pos', [0.8 0.05 btnWdth btnHt], ...
      'background', [0.8 0.8 0.8], ...
      'foreground', [0 0 0], ...
      'string','Close', ...
      'callback', 'viterb(''stop'');', ...
      'visible', 'on');
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(fun,'iterate'),
  
  nIter = varargin{1};
  
  set(hb,'enable','off');
  
  for k=1:nIter,
    
    % Check for convergence
    if all(all(whereBefore == whereMax)),

      % Send message
      xd = get(hl,'xdata'); yd = get(hl,'ydata');
      str = sprintf('Convergence has been reached in %i iterations.\n\nTotal likelihood at the end is %1.2e.', xd(end), yd(end) );
      set(hb(4),'string',str);

      % Send results to base workspace
      assignin('base','viterb_result_means',mmeans);
      assignin('base','viterb_result_vars',vvars);
      assignin('base','viterb_result_priors',ppriors);
      disp(' ');
      disp([ 'VITERB: resulting means, variances and priors are now stored in' ...
	    ' the workspace variables viterb_result_means,' ...
	    ' viterb_result_vars and viterb_result_priors.']);
      break;
      
    else,
      
      % Update clusters and plots
      for i=1:numClust,
	% Classification
	subData = data(whereMax==i,:);
	set( hp(i), 'xdata', subData(:,1), 'ydata',subData(:,2) );
	
	% Update clusters
	mmeans{i} = mean( subData );
	vvars{i} = cov( subData );
	ppriors(i) = size(subData,1) / numPts;
	
	ellip = circle * sqrtm(vvars{i}) + repmat(mmeans{i},100,1);
	set( hc(i,1), 'xdata', ellip(:,1), 'ydata', ellip(:,2) );
	set( hc(i,2), 'xdata', mmeans{i}(1), 'ydata', mmeans{i}(2) );
      end;
	
      whereBefore = whereMax;
      for i=1:numClust,
	% Compute log-likelihood
	x = data - repmat(mmeans{i},numPts,1);
	invSig = inv(vvars{i});
	logLike(i,:) = -0.5 * ( sum( ((x*invSig).*x)' ) );
	logLike(i,:) =  logLike(i,:) - 0.5 * log(det(vvars{i})) + log(ppriors(i));
      end;
      % Find max likelihood
      [maxLike,whereMax] = max(logLike);
      
      % Update likelihood plot
      xd = get(hl,'xdata'); xd = [xd  xd(end)+1];
      yd = get(hl,'ydata'); yd = [yd (sum(maxLike) + piConst)];
      set(hl,'xdata',xd,'ydata',yd);
    
      drawnow;
      
    end;
  
  end;
  
  if all(all(whereBefore == whereMax)),
    set(hb([4 6]),'enable','on');
  else,
    set(hb,'enable','on');
  end;
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(fun,'stop'),
  
  clear data mmeans vvars piConst numClust numPts whereMax whereBefore circle hc hp hb hl cmap;
  close(gcf);
  
else,

  error('Unknown function.');
  
end;
