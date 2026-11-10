function pinn_sanity_check()
% Run this first. Checks that the MATLAB forward pass gives the same numbers
% as PyTorch before any controller uses it.

    net = pinn_load('pinn_weights.json');
    ref = jsondecode(fileread('pinn_reference.json'));

    fprintf('Verifying MATLAB forward pass against PyTorch reference...\n\n');
    maxerr = 0;
    for i = 1:size(ref.x, 1)
        x = ref.x(i, :).';
        [f, g] = pinn_fg(net, x);
        fref = squeeze(ref.f(i, :)).';
        gref = squeeze(ref.g(i, :, :));
        ef = max(abs(f(:) - fref(:)));
        eg = max(abs(g(:) - gref(:)));
        maxerr = max([maxerr, ef, eg]);
        fprintf('  x = [%.4f, %.4f]   |df| = %.2e   |dg| = %.2e\n', ...
                x(1), x(2), ef, eg);
    end
    fprintf('\nOVERALL max error: %.2e\n', maxerr);
    if maxerr < 1e-3
        fprintf('PASS -- MATLAB matches PyTorch. Safe to proceed.\n');
    else
        error('FAIL -- MATLAB does NOT match PyTorch. Do not use in Simulink.');
    end
end