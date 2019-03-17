function f = computeR2(model,data)

%	function f = computeR2(model,data) 

%	
%	<model> is a 1 x N vector of model values 
%	<data> is a 1 x N vector of data points 
%
%	return the coefficient of determination (R^2) 
%	between <model> and <data>.  this value summarizes 
%	how well <model> approximates <data>, and has 
%	an upper bound of 100%. 

    r2 = 100 * (1 - sum((model - data).^2)/sum((model-mean(model)).^2));
    
    f = r2;