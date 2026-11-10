%% setup_nmpc.m
%  Puts everything the Simulink model needs into the base workspace. These
%  used to be set by hand, I scripted them so the runs can be repeated.

clear; clc;

%% Sample time
Ts = 0.01;                    % hr, matches the NMPC step

%% Steady state (the unstable operating point from the paper)
CAs  = 1.9537;                % kmol/m^3
Ts_K = 401.8727;              % K   (named Ts_K so it doesn't clash with Ts)
CA0s = 4;                     % kmol/m^3
Qs   = 0;                     % kJ/hr

%% Initial conditions for the unit delays
yo    = [CAs; Ts_K];          % initial state  [CA; T]
uo    = [CA0s; Qs];           % initial input  [CA0; Q]
upast = uo;                   % previous input
ypast = yo;                   % previous state

%% LSTM state ports: unused by my PINN, but the ports still exist
ho = zeros(10,1);
Co = zeros(10,1);

%% Setpoints
r1 = CAs;                     % CA setpoint
r2 = Ts_K;                    % T  setpoint
y_set = [r1; r2];

%% Disturbance estimate (starts at zero)
dy = [0; 0];

fprintf('Workspace set up. Ts=%g, yo=[%.4f %.2f], uo=[%g %g]\n', ...
        Ts, yo(1), yo(2), uo(1), uo(2));