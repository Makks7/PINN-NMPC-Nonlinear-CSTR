function [u_first_move,fcost,Ecount,exitflag,ypred,hk,Ck,xPx,Tconst] = ucompute_PINN(...
    r1,r2,u1set,u2set,Wy1,Wy2,Wu1,Wu2,Wdu1,Wdu2,xo,uo,upast,ypast,Np,Nu,dy...
    ,x1min,x2min,x1max,x2max,du1min,du2min,du1max,du2max...
    ,U_initial,UMIN_colvector,UMAX_colvector,ho,Co,P,alpha)
%
% Same signature as Dr. Bamimore's original ucompute, so nothing in the
% Simulink diagram needs rewiring. Only the prediction model inside changed.
% upast, ypast, ho and Co still come in but the PINN does not use them, it
% only needs the current state and the candidate inputs. ho and Co are
% passed straight through so the feedback loops in the diagram stay intact.

dt = 0.01 ;      % NMPC step (hr)

OPTIONS = optimoptions('fmincon','Algorithm','sqp','Display','off', ...
    'FiniteDifferenceStepSize',1e-4,'FiniteDifferenceType','central', ...
    'MaxIterations',60);

tic
[uopt,fval,exitflag] = fmincon( ...
    @(u) obj(u,r1,r2,u1set,u2set,Wy1,Wy2,Wu1,Wu2,Wdu1,Wdu2,xo,uo,Np,Nu,dy,dt), ...
    U_initial,[],[],[],[],UMIN_colvector,UMAX_colvector, ...
    @(u) nlfun(u,r1,r2,uo,xo,Np,Nu,x1min,x2min,x1max,x2max, ...
               du1min,du2min,du1max,du2max,dy,P,alpha,dt), OPTIONS);
Ecount = toc ;

u1_first_move = uopt(1,:) ;
u2_first_move = uopt(Nu+1,:) ;
u_first_move  = [u1_first_move ; u2_first_move] ;
fcost = fval ;

% one-step prediction I report back to the diagram
yk1   = PINN_output_prediction(xo,uo,uopt,Nu,Np,dt) ;
ypred = (yk1(1,:) + dy')' ;

% I keep the LSTM state ports alive but unused, so nothing downstream breaks.
hk = ho ;  Ck = Co ;

[xPx,Tconst] = terminal_const(uopt,xo,uo,Np,Nu,dy,r1,r2,P,alpha,dt);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cost = obj(u,r1,r2,u1set,u2set,Wy1,Wy2,Wu1,Wu2,Wdu1,Wdu2,xo,uo,Np,Nu,dy,dt)

yk = output_prediction(u,xo,uo,Np,Nu,dy,dt);

u1 = u(1:Nu);  u2 = u(Nu+1:2*Nu);
du1(1) = u1(1)-uo(1);  du2(1) = u2(1)-uo(2);
for k = 2:Nu
    du1(k) = u1(k)-u1(k-1);  du2(k) = u2(k)-u2(k-1);
end

fy1 = (r1*ones(Np,1) - yk(:,1));      % tracking error, state 1
fy2 = (r2*ones(Np,1) - yk(:,2));      % tracking error, state 2
fu1 = (u1set*ones(Nu,1) - u1(1:Nu));  % input effort
fu2 = (u2set*ones(Nu,1) - u2(1:Nu));
fdu1 = du1';  fdu2 = du2';            % move suppression

cost = fy1'*Wy1*fy1 + fy2'*Wy2*fy2 ...
     + fu1'*Wu1*fu1 + fu2'*Wu2*fu2 ...
     + fdu1'*Wdu1*fdu1 + fdu2'*Wdu2*fdu2 ;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [c,ceq] = nlfun(u,r1,r2,uo,xo,Np,Nu,x1min,x2min,x1max,x2max, ...
                         du1min,du2min,du1max,du2max,dy,P,alpha,dt)

yk = output_prediction(u,xo,uo,Np,Nu,dy,dt);

u1 = u(1:Nu);  u2 = u(Nu+1:2*Nu);
du1(1) = u1(1)-uo(1);  du2(1) = u2(1)-uo(2);
for k = 2:Nu
    du1(k) = u1(k)-u1(k-1);  du2(k) = u2(k)-u2(k-1);
end

y1k = yk(:,1);  y2k = yk(:,2);

ineq_y1_min = x1min*ones(Np,1) - y1k;
ineq_y1_max = y1k - x1max*ones(Np,1);
ineq_y2_min = x2min*ones(Np,1) - y2k;
ineq_y2_max = y2k - x2max*ones(Np,1);
ineq_du1_min = du1min*ones(Nu,1) - du1';
ineq_du1_max = du1' - du1max*ones(Nu,1);
ineq_du2_min = du2min*ones(Nu,1) - du2';
ineq_du2_max = du2' - du2max*ones(Nu,1);

%%%%%%%%%%%%%%%%%% LYAPUNOV CONSTRAINT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% I kept my supervisor's structure, but I take xdot from the PINN DIRECTLY
% (xdot = f_nn + g_nn*u) instead of finite-differencing the prediction. The
% PINN is a derivative model, so this is the exact V_dot instead of an
% approximation of it: which is the quantity the stability proof needs.
g = [] ;
rhomin = alpha ;
kgain  = 15 ;

y_set = [r1; r2];
xt = xo(:) - y_set ;
V  = xt' * P * xt ;

u_now = [u1(1); u2(1)] ;                  % the move actually applied now
xdot  = pinn_xdot_pub(xo(:), u_now) ;     % exact derivative from the PINN
V_dot = 2 * xt' * P * xdot ;

if V > rhomin
    g = [g ; V_dot + kgain*abs(V/100)] ;  % must drive V down
else
    xNN   = yk(end,:)' - y_set ;
    V_xNN = xNN' * P * xNN ;
    g = [g ; V_xNN - rhomin] ;            % must stay inside the small set
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c = [ineq_y1_min; ineq_y1_max; ineq_du1_min; ineq_du1_max; ...
     ineq_y2_min; ineq_y2_max; ineq_du2_min; ineq_du2_max ; g] ;
ceq = [] ;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xPx,Tconst] = terminal_const(u,xo,uo,Np,Nu,dy,r1,r2,P,alpha,dt)
yk = output_prediction(u,xo,uo,Np,Nu,dy,dt);
e  = yk(Np,:) - [r1,r2] ;
xPx = e*P*e' ;
Tconst = xPx - alpha ;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function yk = output_prediction(u,xo,uo,Np,Nu,dy,dt)
% I add the disturbance estimate dy, as the LSTM version did.
y_k = PINN_output_prediction(xo,uo,u,Nu,Np,dt) ;
yk  = y_k + repmat(dy',Np,1) ;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function xdot = pinn_xdot_pub(x,u)
persistent NET
if isempty(NET), NET = pinn_load('pinn_weights.json'); end
[f,g] = pinn_fg(NET,x);
xdot = f + g*u(:);
end