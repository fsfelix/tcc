function [] = emalgo(fun,varargin)

% EMALGO EM algorithm explorer
%
%   Launch it with EMALGO(DATA,NCLUST) where DATA is the matrix
%   of observations (one observation per row) and NCLUST is the
%   desired number of clusters.
%
%   The clusters are initialized with a heuristic that spreads
%   them randomly around mean(DATA) with standard deviation
%   sqrtm(cov(DATA)*10). Their initial covariance is set to cov(DATA).
%
%   If you want to set your own initial clusters, use
%   EMALGO(DATA,MEANS,VARS) where MEANS and VARS are cell arrays
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
%     emalgo(data,means,vars);
%


global data mmeans vvars piConst ppriors weights sumWeights numClust numPts dim circle hc hp hb hl cmap;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isstr(fun),
  
  data = fun;
  [numPts,dim] = size(data);
  
  % Init clusters and likelihood
  if iscell(varargin{1}),
    
    mmeans = varargin{1};
    vvars = varargin{2};
    
    numClust = length(varargin{1});
    
    if (length(varargin)<3),
      ppriors(1:numClust) = 1 / numClust;
    else,
      ppriors = varargin{3};      
    end;    
    
  else,
    
    numClust = varargin{1};

    startMean = mean(data);
    startvvars = cov(data);
    startDev = sqrtm(startvvars * 10);
    for i=1:numClust,
      mmeans{i} = randn(1,dim) * startDev + startMean;
      vvars{i} = startvvars;
      ppriors(i) = 1 / numClust;
    end;
  
  end;
  
  % Make colormap for classification
  cmap = hsv(numClust);
    
  % Define normalization constant
  piConst = (2*pi)^(-dim/2);
  %%%%%%%%%
  %% E step for next iteration
  % Update weights
  for i=1:numClust,
    % Compute likelihood
    x = data - repmat(mmeans{i},numPts,1);
    invSig = inv(vvars{i});
    likelihood(i,:) = piConst * (1/sqrt(det(vvars{i}))) ...
	* exp( -0.5 * sum( ((x*invSig).*x)' ) ) * ppriors(i);
    
  end;
  weights = ( likelihood ./ repmat( sum(likelihood) , numClust , 1 ) )';
  sumWeights = sum(weights);

  % Init figure
  hf = figure('name','EM Algorithm Explorer','numbertitle','off');
  
  % Attribution plot
  subplot('position',[ 0 0.05 0.45 0.9 ]);
  plot(data(:,1),data(:,2),'y+');
  set(gca,'xlim',[0 1000],'ylim',[0 3000],'dataaspectratio',[1 1 1], ...
      'drawmode','fast');
  xlabel('F1 (Hz)'); ylabel('F2 (Hz)');
  grid on; hold on;
  
  circle = [cos(linspace(-pi, pi, 100)') sin(linspace(-pi, pi, 100)')];
  for i=1:numClust,
    ellip = circle * sqrtm(vvars{i}) + repmat(mmeans{i},100,1);
    hc(i,1) = line(ellip(:,1),ellip(:,2),10*ones(size(ellip,1),1), ...
	'color',[1 0 0],'linew',2);
    hc(i,2) = line(mmeans{i}(1),mmeans{i}(2), 10, ...
	'marker','+','markersize',10,'color',[1 0 0],'linew',2);
  end;
  
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
      'callback', 'emalgo(''iterate'',1);', ...
      'visible', 'on');
  
  hb(2) = uicontrol('style','push', ...
      'units','norm', ...
      'pos', [0.65 0.3 btnWdth btnHt], ...
      'background', [0.8 0.8 0.8], ...
      'foreground', [0 0 0], ...
      'string','Iterate 5 times', ...
      'callback', 'emalgo(''iterate'',5);', ...
      'visible', 'on');
  
  hb(3) = uicontrol('style','push', ...
      'units','norm', ...
      'pos', [0.8 0.3 btnWdth btnHt], ...
      'background', [0.8 0.8 0.8], ...
      'foreground', [0 0 0], ...
      'string','Iterate 10 times', ...
      'callback', 'emalgo(''iterate'',10);', ...
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
      'string','Iterate 100 times', ...
      'callback', 'emalgo(''iterate'',100);', ...
      'visible', 'on');
  
  cstr = [ ...
	'global vvars mmeans ppriors;' ...
	'assignin(''base'',''em_resultmmeans'',mmeans);' ...
	'assignin(''base'',''em_resultvvars'',vvars);' ...
	'assignin(''base'',''em_result_priors'',ppriors);' ...
	'disp('' '');' ...
	'disp(''EMALGO: resulting means, variances and priors are now stored in' ...
	' the workspace variables em_result_means, em_result_vars ' ...
	'and em_result_priors.'');'  ...
	];
  hb(7) = uicontrol('style','push', ...
      'units','norm', ...
      'pos', [0.65 0.05 btnWdth btnHt], ...
      'background', [0.8 0.8 0.8], ...
      'foreground', [0 0 0], ...
      'string','Save clusters', ...
      'callback', cstr, ...
      'visible', 'on');
  
  hb(6) = uicontrol('style','push', ...
      'units','norm', ...
      'pos', [0.8 0.05 btnWdth btnHt], ...
      'background', [0.8 0.8 0.8], ...
      'foreground', [0 0 0], ...
      'string','Close', ...
      'callback', 'emalgo(''stop'');', ...
      'visible', 'on');
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(fun,'iterate'),
  
  nIter = varargin{1};
  
  set(hb,'enable','off');
    
  for k=1:nIter,

    %%%%%%%%%
    %% M step
    % Update clusters
    for i=1:numClust,
      W = repmat(weights(:,i),1,dim);
      mmeans{i} = sum(data.*W) / sumWeights(i);
      x = ( data - repmat(mmeans{i},numPts,1) );
      vvars{i} = ((x.*W)' *  x) / sumWeights(i) ;
      
      ellip = circle * sqrtm(vvars{i}) + repmat(mmeans{i},100,1);
      set( hc(i,1), 'xdata', ellip(:,1), 'ydata', ellip(:,2) );
      set( hc(i,2), 'xdata', mmeans{i}(1), 'ydata', mmeans{i}(2) );
    end;
    ppriors = sumWeights / numPts;

    %%%%%%%%%
    %% E step for next iteration
    % Update weights
    for i=1:numClust,
      % Compute likelihood
      x = data - repmat(mmeans{i},numPts,1);
      invSig = inv(vvars{i});
      likelihood(i,:) = piConst * (1/sqrt(det(vvars{i}))) ...
	  * exp( -0.5 * sum( ((x*invSig).*x)' ) ) * ppriors(i);
	  
    end;
    totLike = sum(log(sum(likelihood)));
    weights = ( likelihood ./ repmat( sum(likelihood) , numClust , 1 ) )';
    sumWeights = sum(weights);
    
    %%%%%%%%%
    % Update likelihood plot
    xd = get(hl,'xdata'); xd = [xd  xd(end)+1];
    yd = get(hl,'ydata'); yd = [yd totLike];
    set(hl,'xdata',xd,'ydata',yd);
    
    drawnow;
    
  end;

  set(hb,'enable','on');
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(fun,'stop'),
  
  clear data mmeans vvars piConst ppriors weights sumWeights numClust numPts dim circle hc hp hb hl cmap;
  close(gcf);
  
else,

  error('Unknown function.');
  
end;
