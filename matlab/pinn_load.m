function net = pinn_load(jsonfile)
% Loads the exported PINN weights (JSON) into a MATLAB struct.
% Usage: net = pinn_load('pinn_weights.json');
    txt = fileread(jsonfile);
    net = jsondecode(txt);
end