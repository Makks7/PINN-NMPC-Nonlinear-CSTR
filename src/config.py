"""config.py - all the constants and settings in one place, so nothing is typed twice."""

import os

# I resolve paths relative to this file so it runs from anywhere.
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "outputs")
FIG_DIR  = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# I train on BOTH of my supervisor's datasets: the open-loop run and the
# closed-loop (PID) run. Listing them here is the only place I choose data.
# preprocess.py finite-differences each file separately so I never
# differentiate across the join between two different experiments.
# I train on all three of the group's datasets. Evaluated fairly on a common
# held-out test set drawn from all three sources, the combination beats any
# subset (R2 0.962/0.952 vs 0.894/0.918 for open+closed and 0.854/0.870 for
# pulses alone) and gives the lowest g-recovery error. The pulse runs are a
# added on purpose: they excite concentration-direction dynamics that the
# real plant/controller data almost never visits, so they fill a gap rather
# than replace the real data.
DATA_FILES = [
    os.path.join(DATA_DIR, "cstr_data.csv"),             # open loop
    os.path.join(DATA_DIR, "cstr_data_closed (2).csv"),  # closed loop (PID)
    os.path.join(DATA_DIR, "cstr_data_pulses.csv"),      # pulse supplement (400 trajectories)
]

# I use the group's column convention: u1=CA0, u2=Q, y1=CA, y2=T.
COL_INPUTS = ["u1", "u2"]
COL_STATES = ["y1", "y2"]
DT         = 0.01
ROLLOUT_DT = 0.001   # control update step for closed-loop rollout

# I took the CSTR parameters straight from the paper (Eq. 38) / ode_set_open.m.
T0   = 300.0
F    = 5.0
V    = 1.0
E    = 5.0e4
K0   = 8.46e6
DH   = -1.15e4
CP   = 0.231
R    = 8.314
RHOL = 1000.0

# I build deviation variables around this unstable steady state.
CAS  = 1.9537
TS   = 401.8727
CA0S = 4.0
QS   = 0.0

# I use the paper's Lyapunov function V = 1060*x1^2 + 44*x1*x2 + 0.52*x2^2.
P_MATRIX = [[1060.0, 22.0],
            [22.0,   0.52]]
C1_LYAP  = 1.0
RHO_S    = 20.0

# I saturate the Sontag controller to the paper's input constraints.
DU_MAX = [3.5, 5.0e5]           # deviation-input bounds |du| <= DU_MAX
U_LO = [CA0S - 3.5, QS - 5.0e5]  # absolute box, kept for reference
U_HI = [CA0S + 3.5, QS + 5.0e5]

# I normalise internally because dT/dt ~ 1e3 and Q ~ 1e5 wreck the optimiser.
XDOT_SCALE = [10.0, 2000.0]
U_SCALE    = [3.5,  5.0e5]

# I reused my pH-PINN backbone.
HIDDEN_LAYERS = [128, 128, 64]
ACTIVATION    = "tanh"

# I train with Adam + early stopping, same as before.
SEED           = 42
TEST_FRAC      = 0.15
VAL_FRAC       = 0.15
BATCH_SIZE     = 256
MAX_EPOCHS     = 2000
LEARNING_RATE  = 1e-3
LR_DECAY_STEP  = 200
LR_DECAY_GAMMA = 0.5
PATIENCE       = 50
GRAD_CLIP      = 1.0

# I weight the three loss terms here.
LAMBDA_PHYS = 1.0
LAMBDA_G    = 20.0
LAMBDA_LYAP = 0.02