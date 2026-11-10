function dist = Dist_rejector(r_k,W_dist,y_plant)
% Compares the plant output with what the model predicted and returns the
% difference. The NMPC uses this as the disturbance estimate.
e_k = y_plant - r_k;

% error_min = -10; %Minimum permissible error
% error_max = +10; %Maximum permissible error

error_min = -20; %Minimum permissible error
error_max = +20; %Maximum permissible error


error_func = zeros(2,1);
dist = zeros(2,1);

if e_k(1) > error_min && e_k(1) < error_max 
    % error_func(1) = tanh(e_k(1));
    error_func(1) = (e_k(1));
else
    error_func(1) = 0;
end

if e_k(2) > error_min && e_k(2) < error_max 
    % error_func(2) = tanh(e_k(2));
    error_func(2) = (e_k(2));
else
    error_func(2) = 0;
end


%The disturbance function is a linear function of the error function.

dist(1) = (error_func(1)*W_dist(1)) ;
dist(2) = (error_func(2)*W_dist(2)) ;

