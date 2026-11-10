# PINN-NMPC for a Non-Isothermal CSTR

This repository holds the code, data and trained models for the **non-isothermal CSTR case study** of the PINN-NMPC framework. The reactor is open-loop unstable. A Lyapunov-stable, control-affine neural network is trained with physics-informed and stability terms in the loss. It is then used in two controllers: a Sontag-formula controller, and a Lyapunov-based NMPC in MATLAB/Simulink. Both hold the reactor at an operating point from which the uncontrolled plant drifts away.

This is the third of five benchmark case studies in the final-year thesis *Physics-informed neural networks for nonlinear model predictive control* (Department of Chemical Engineering, Obafemi Awolowo University, 2026). The other four are an isothermal CSTR, pH neutralization, a cascaded three-tank system and a continuous fermentor.

## The process

A jacketed CSTR with an irreversible, exothermic, second-order reaction A → B (Tan et al., 2026; Wu et al., 2019).

- **States:** concentration of A, C_A, and reactor temperature, T
- **Inputs:** feed concentration C_A0 and rate of heat input Q
- **Operating point:** C_As = 1.9537 kmol/m³, T_s = 401.87 K (C_A0s = 4 kmol/m³, Q_s = 0), open-loop unstable
- **Input bounds:** 0.5 ≤ C_A0 ≤ 7.5 kmol/m³, −5×10⁵ ≤ Q ≤ 5×10⁵ kJ/h

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| F | 5 m³/h | V | 1 m³ |
| k₀ | 8.46×10⁶ m³/(kmol·h) | E | 5×10⁴ kJ/kmol |
| R | 8.314 kJ/(kmol·K) | ΔH | −1.15×10⁴ kJ/kmol |
| ρ_L | 1,000 kg/m³ | C_p | 0.231 kJ/(kg·K) |
| T₀ | 300 K | | |

## The model

The network has the same control-affine structure as the plant, dz/dt = f(z) + g(z)u.

- Shared backbone of three hidden layers (128, 128, 64 neurons, tanh), feeding an f-head (2 outputs) and a g-head (2×2 input matrix), 25,542 parameters in total
- Predicts continuous-time derivatives, not the next sampled state
- Loss: L = L_data + λ_p L_phys + λ_g L_g + λ_V L_V, with λ_p = 1, λ_g = 20 and λ_V = 0.02
- Lyapunov function V(z) = zᵀPz with P = [1060, 22; 22, 0.52], stability region ρ = 368, terminal set ρ_s = 20
- Controller from the learned f and g by Sontag's universal formula
- Adam, learning rate 10⁻³ halved every 200 epochs, batch size 256, gradient clipping at 1.0, early stopping (patience 50, at most 2,000 epochs)

The term L_g, which ties the learned input matrix to its analytic value, is not in the source method of Tan et al. (2026). It was added in this study. Without it, the network fitted the data closely and passed its own Lyapunov check, but the controller failed on the true plant because the learned input matrix was wrong. This is a closed-loop identifiability problem. For that reason, every stability check here is done on the **true plant**, not on the model.

## Data

- 45,600 samples at Δt = 0.01 h from three records: a 100 h open-loop run (10,000), a 120 h PI closed-loop run (12,000) and 400 short trajectories near the operating point (23,600)
- Random 70/15/15 split: 31,920 for training, 6,840 for validation and 6,840 for testing
- Data generated in Simulink and passed to Python as data files

## Results

| Index | C_A channel | T channel |
|---|---|---|
| Derivative prediction, test R² | 0.9895 | 0.9892 |
| Derivative prediction, test RMSE | 0.476 kmol m⁻³ h⁻¹ | 21.5 K h⁻¹ |
| Free-run state prediction, R² | 0.9991 | 0.9991 |
| Free-run state prediction, RMSE | 0.0089 kmol m⁻³ | 0.65 K |
| Relative error in recovered input gain | 5.8% (g₁₁) | 3.4% (g₂₂) |

| Stability and control index | Value |
|---|---|
| Grid states with dV/dt < 0 on the true plant, inside Ω_ρ | 2,110 of 2,110 |
| Grid states with dV/dt < 0 on the true plant, full grid | 12,655 of 12,656 (99.99%) |
| Sontag closed loop from four benchmark initial conditions | All converge, final V = 11.1 to 13.2 (below ρ_s = 20) |
| Lyapunov-based NMPC run (2 h, from C_A = 0.95 kmol/m³, T = 448 K) | IAE = 12.89 |
| Settling of the NMPC run | C_A within 2% by 1.37 h, T within 1 K by 1.43 h |
| Longest single optimization | 2.9 s (sampling period 36 s) |

**Data requirement.** Retraining on subsets of the training data (three repeats per size) showed that about **5,000 samples**, or 50 h of operation, are needed. Below that, the temperature derivative is not learned (median R² ≤ 0.06).

| Training samples | R² dC_A/dt | R² dT/dt | g₁₁ error (%) | g₂₂ error (%) | States with dV/dt ≥ 0 |
|---|---|---|---|---|---|
| 500 | 0.45 | −0.05 | 26.8 | 19.9 | 36 |
| 1,000 | 0.49 | 0.03 | 13.0 | 16.9 | 27 |
| 2,500 | 0.51 | 0.06 | 11.3 | 12.1 | 25 |
| 5,000 | 0.83 | 0.77 | 10.1 | 8.8 | 1 |
| 10,000 | 0.97 | 0.97 | 7.8 | 3.5 | 1 |
| 20,000 | 0.98 | 0.98 | 7.1 | 3.6 | 1 |
| 31,920 (full) | 0.98 | 0.98 | 5.7 | 2.9 | 1 |

*Note:* the Simulink NMPC loads an earlier trained network (exported 19 July 2026) with the same architecture. It passed the same checks: input gains within 0.96% and 0.57%, dV/dt < 0 at 12,655 of 12,656 grid states, and all four Sontag runs converge.

## Lyapunov-based NMPC settings

| Setting | Value |
|---|---|
| Sampling period, horizons | Δt = 0.01 h, N_p = 10, N_u = 2 |
| Setpoint | C_A = 1.95 kmol/m³, T = 402 K |
| Output weights | w_y = (1.026, 4.98×10⁻³) |
| Move weights | w_Δu = (0.25, 10⁻¹⁰) |
| Lyapunov constraint | dV/dt ≤ −0.15 V while V > 20; terminal V ≤ 20 |
| Integral term | W = diag(0.06, 0.075) |
| Solver | fmincon (SQP), central finite differences, at most 60 iterations |
| Prediction inside the controller | Fourth-order Runge-Kutta on f_θ + g_θ u |
| Plant | Mass and energy balances integrated with ode15s |

## Workflow

1. **Simulink:** generate the excitation data.
2. **Python (PyTorch):** train the control-affine network, check the Lyapunov condition on the true plant, and run the sample-size study.
3. **Export:** the trained weights, biases and scaling ranges are saved to a single `.mat` file.
4. **MATLAB/Simulink:** a short MATLAB function reads the `.mat` file and does the forward pass, so no Python runtime is needed during closed-loop simulation. The Lyapunov-based NMPC is then run with fmincon.

## Software

| Tool | Version |
|---|---|
| PyTorch | 2.x |
| Matplotlib | 3.10.8 |
| MATLAB, Simulink, Optimization Toolbox | R2025b |

The runs were made on Windows with an Intel Core i7-7600U processor at 2.80 GHz.

## Acknowledgement

The Simulink implementation of the Lyapunov-based NMPC for this reactor was provided by the supervisor, Dr. A. Bamimore. In this work, the network was inserted through an output-prediction function and a replacement control routine, and the Lyapunov matrix P was updated.

## References

- Tan, W. G. Y., Xiao, M., Wu, G. and Wu, Z. (2026). Lyapunov-stable neural networks for modelling and control of nonlinear systems. *International Journal of Control*, 99(2), 578–594. https://doi.org/10.1080/00207179.2025.2525544
- Sontag, E. D. (1989). A 'universal' construction of Artstein's theorem on nonlinear stabilization. *Systems and Control Letters*, 13(2), 117–123.
- Wu, Z., Tran, A., Rincon, D. and Christofides, P. D. (2019). Machine learning-based predictive control of nonlinear processes. Part I: Theory. *AIChE Journal*, 65(11), e16729.

## License

MIT License. See `LICENSE`.
