function new_vector = bootstrap(vector)
%BOOTSTRAP Summary of this function goes here
%   Detailed explanation goes here

% Take an randomized indexes to query from the original sample
ix = ceil(length(vector)* rand(1, length(vector)));

% Return a vector chosen from the original one, but at random places
new_vector = vector(ix);

end