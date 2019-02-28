function yy = nnregress(xx,x,y)

%	function yy = nnregress(xx,x,y) 

%	
%	<xx> is a vector of x-coordinates to evaluate at 
%	<x> is a vector of x-data 
%	<y> is a vector of y-data 
%
%	return a vector with the predictions of a 
%	nearest-neighbor regression model evaluated 
%	at the values in <xx>. 

results = zeros(size(xx));

% for each uncertain point
for i=1:length(xx)
    % calculate Euclidian distance
    distances = abs(xx(i) - x);
    
    % find an index of the smallest distance
    min_dist = min(distances);
    
    index = distances==min_dist;
    
    
    results(i) = y(index);
    
end

yy = results;