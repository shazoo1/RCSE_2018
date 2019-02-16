function new_vector = ex4_modified_bootstrap(vector, num_samples)
%BOOTSTRAP Summary of this function goes here
%   Detailed explanation goes here


input_size = length(vector);

% Take an randomized indexes to query from the original sample
%ix = randsample(input_size,input_size,true);

ix = ceil(input_size* rand(num_samples, input_size));



% Return a vector chosen from the original one, but at random places
new_vector = vector(ix);

end