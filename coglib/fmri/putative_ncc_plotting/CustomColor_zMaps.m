% Define the starting and ending colors
start_color = [255,0,0] ./ 255;  % Normalize to range [0,1]
end_color = [255,255,0] ./ 255;  % Normalize to range [0,1]

% Create the colormap by linearly interpolating between the two colors
A = interp1([1,256], [start_color; end_color], 1:256);

% Define the starting and ending colors
start_color = [0,0,255] ./ 255;  % Normalize to range [0,1]
end_color = [0, 255, 255] ./ 255;  % Normalize to range [0,1]

% Create the colormap by linearly interpolating between the two colors
D = interp1([1,256], [start_color; end_color], 1:256);
