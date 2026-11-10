function y_k = PINN_output_prediction(xo, uo, uu, Nu, Np, dt)
% PINN_output_prediction
% Rolls the PINN forward Np steps to give the NMPC its predictions. It takes
% the place of LSTM_output_prediction in the original setup.
%
%   xo : 2x1 current state [CA; T], measured, absolute units
%   uo : 2x1 last applied input (only used for the first move)
%   uu : decision vector from the optimiser [u1(1..Nu); u2(1..Nu)]
%   Nu : control horizon, Np : prediction horizon, dt : step size (hr)
%   y_k: Np x 2 predicted states
%
% The LSTM needed past inputs, past outputs and a hidden state. The PINN
% does not, because it is control-affine and only depends on x:
%       xdot = f_nn(x) + g_nn(x)*u
% So I just integrate it forward with RK4, one step per prediction point.

    persistent NET
    if isempty(NET)
        NET = pinn_load('pinn_weights.json');   % load once, reuse every call
    end

    u1 = uu(1:Nu);
    u2 = uu(Nu+1:2*Nu);

    y_k = zeros(Np, 2);
    x   = xo(:);

    for i = 1:Np
        % I hold the last optimised move beyond the control horizon.
        if i <= Nu
            u = [u1(i); u2(i)];
        else
            u = [u1(Nu); u2(Nu)];
        end

        x = rk4_pinn(NET, x, u, dt);
        y_k(i, :) = x.';
    end
end

function xn = rk4_pinn(net, x, u, dt)
% I integrate one step with RK4 using the PINN's own derivative.
    k1 = pinn_xdot(net, x,             u);
    k2 = pinn_xdot(net, x + .5*dt*k1,  u);
    k3 = pinn_xdot(net, x + .5*dt*k2,  u);
    k4 = pinn_xdot(net, x + dt*k3,     u);
    xn = x + dt/6*(k1 + 2*k2 + 2*k3 + k4);
end

function xdot = pinn_xdot(net, x, u)
% xdot = f_nn(x) + g_nn(x)*u
    [f, g] = pinn_fg(net, x);
    xdot = f + g*u(:);
end