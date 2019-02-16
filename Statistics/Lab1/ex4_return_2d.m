function result = ex4_return_2d(sample_size,num_count)
%EX4_RETURN_2D Summary of this function goes here
%   Detailed explanation goes here

x = rand(1,sample_size);

result = ex4_modified_bootstrap(x, num_count);

end

