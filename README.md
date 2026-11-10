# Non-isothermal CSTR (case study 3)

PINN in control-affine form (xdot = f_nn(x) + g_nn(x) u) for the nonlinear
CSTR, trained with a Lyapunov stability term (LSNN) and then used inside the
NMPC in Simulink.

- `src/` - Python: config, data preprocessing, model, physics, losses,
  training, testing, validation, the sample-size study and the figures
- `matlab/` - MATLAB/Simulink: loading the exported weights, the PINN forward
  pass, the prediction over the horizon and the NMPC (ucompute_PINN.m)
- `data/`, `outputs/`, `figures/` - dataset, trained model and results

Order: `python src/train.py`, `python src/test.py`, `python src/validate.py`,
`python src/exports.py`, then in MATLAB `pinn_sanity_check.m` and
`setup_nmpc.m` before running the Simulink model.
