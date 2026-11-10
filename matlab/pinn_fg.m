function [f, g] = pinn_fg(net, x)
% Same forward pass as the PyTorch model, gives f_nn(x) and g_nn(x).
% x is the 2x1 physical state [CA; T]. f is 2x1 and g is 2x2.

    xmin = net.xscaler_min(:); xmax = net.xscaler_max(:);
    h = (x(:) - xmin) ./ (xmax - xmin);

    for k = 1:numel(net.backbone)
        W = net.backbone(k).W; b = net.backbone(k).b(:);
        h = tanh(W*h + b);
    end

    f_hat = net.f_head.W * h + net.f_head.b(:);
    g_hat = net.g_head.W * h + net.g_head.b(:);
    % PyTorch view() is row-major, MATLAB reshape() is column-major -
    % reshape into the transposed shape then transpose back to match.
    g_hat = reshape(g_hat, [net.n_input, net.n_state]).';

    xs = net.xdot_scale(:); us = net.u_scale(:);
    f = f_hat .* xs;
    g = g_hat .* xs ./ us.';
end