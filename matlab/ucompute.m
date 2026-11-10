function [u_first_move,fcost,Ecount,exitflag,ypred,hk,Ck,xPx, Tconst]=ucompute(r1,r2,u1set,u2set,Wy1,Wy2...
    ,Wu1,Wu2,Wdu1,Wdu2,xo,uo,upast,ypast,Np,Nu,dy...
    ,x1min,x2min,x1max,x2max,du1min,du2min,du1max,du2max...
    ,U_initial,UMIN_colvector,UMAX_colvector,ho,Co,P,alpha)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FD_tol=1e-5 ;
% FD_tol=1e-3 ;
FD_tol=1e-2 ;
% OPTIONS = optimset('Algorithm','sqp','Display','final');
OPTIONS = optimoptions('fmincon','Algorithm','sqp','Display','final', 'FiniteDifferenceStepSize', FD_tol,'FiniteDifferenceType', 'central');
% OPTIONS = optimoptions('fmincon','Algorithm','sqp','Display','final', 'FiniteDifferenceStepSize', FD_tol,'FiniteDifferenceType', 'central');
% OPTIONS = optimoptions('fmincon','Algorithm','sqp','Display','final', 'FiniteDifferenceStepSize', 1e-5);
% OPTIONS = optimoptions('fmincon','Algorithm','sqp','Display','final', 'FiniteDifferenceStepSize', 1e-1);
% OPTIONS = optimset('Algorithm','active-set');
% OPTIONS = optimset('Algorithm','interior-point');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % OPTIONS = optimset('Display','off','MaxIter',1E30,'TolCon',1e-8,'Algorithm','active-set');
   

tic

[uopt,fval,exitflag] =fmincon(@(u) obj(u,r1,r2,u1set,u2set,Wy1,Wy2...
    ,Wu1,Wu2,Wdu1,Wdu2,upast,ypast,xo,uo,Np,Nu,dy,ho,Co)...
    ,U_initial,[],[],[],[],UMIN_colvector,UMAX_colvector,@(u) nlfun(u,r1,r2,uo,xo,upast,ypast,Np,Nu...
    ,x1min,x2min,x1max,x2max,du1min,du2min,du1max,du2max,dy,ho,Co,P,alpha),OPTIONS)   ;





%[uopt,fval] =fmincon(@(u) obj(r1,r2,u1set,u2set,Wy1,Wy2,Wu1,Wu2,Wdu1,Wdu2,u,xo,uo,h,Np,Nu),U_initial,[],[],[],[],UMIN,UMAX)
% [X,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN]
 Ecount = toc ;
uopt
                     
u1_first_move=uopt(1,:)      ;
u2_first_move=uopt(Nu+1,:)   ;
% du3_first_move=duopt(2*Nu+1,:)   ;

u_first_move=[u1_first_move ;u2_first_move ]   ;
fcost=fval ;
% u_1=uo ; u_2=upast ;    xt=[u_1 ; u_2] ;
% yk=output_prediction(du,xo,uo,upast,Np,Nu,dy,update_state);

%[u1_1;  u2_1 ; u1_2 ;  u2_2 ];
% [ypred, hk, Ck] = lstm_network(xt,ho,Co) ;
% [ypred,hk,Ck] = LSTM_output_prediction(xo,uo,uopt,upast,Nu,Np,ho,Co) ;
% [ypred,hk,Ck] = LSTM_output_prediction(xo,uo,uopt,upast,1,1,ho,Co) ;
% ypred = LSTM_output_prediction(xo,uo,uopt,upast,1,1,true)
% ypred = LSTM_output_prediction(xo,uo,u_first_move,upast,ypast,1,1,true) ;


[ypred,hk,Ck] = LSTM_output_prediction(xo,uo,u_first_move,upast,ypast,1,1,ho,Co) ;
ypred=double(ypred) ; hk=double(hk) ; Ck=double(Ck)  ;

[xPx, Tconst]=terminal_const(uopt,xo,uo,upast,ypast,Np,Nu,dy,ho,Co,r1,r2,P,alpha);
% [xPx, Tconst]=terminal_const(uopt,xo,uo,upast,ypast,Np,Nu,[0;0],ho,Co,r1,r2,P,alpha);

% ypred=double(ypred) ; hk=double(hk) ; Ck=double(Ck)  ;

% ypred = output_prediction(du_first_move,xo,uo,upast,1,1,0,true);

end


function cost=obj(u,r1,r2,u1set,u2set...
           ,Wy1,Wy2,Wu1,Wu2,Wdu1,Wdu2...
           ,upast,ypast,xo,uo,Np,Nu,dy,ho,Co);

% yk=output_prediction(du,xo,uo,upast,Np,Nu,dy,false);
% yk=output_prediction(u,xo,uo,upast,ypast,Np,Nu,dy) ;
yk=output_prediction(u,xo,uo,upast,ypast,Np,Nu,dy,ho,Co);



% du1 = zeros(Nu,1); 
% du2 = zeros(Nu,1);  
% du3 = zeros(Nu,1);

% u1 = zeros(Nu,1); 
% u2 = zeros(Nu,1);  
% u3 = zeros(Nu,1);

% 
% du1 = zeros(Nu,1); 
% du2 = zeros(Nu,1);  
% 
% 
% 
% u1 = zeros(Nu,1); 
% u2 = zeros(Nu,1);  


u1 = u(1:Nu);
u2 = u(Nu+1:2*Nu);



%%%%%%%%%%%%%%%%%%%%%%%%
    du1(1)=u1(1)- uo(1)  ;   du2(1)=u2(1)- uo(2) ;    
for k=2:Nu
    du1(k)= u1(k)- u1(k-1)  ;  du2(k)= u2(k)- u2(k-1) ;
end


% OUTPUT COST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

r1set=r1  ;  r2set=r2   ; 

fy1=(r1set*ones(Np,1)-yk(:,1));
fy2=(r2set*ones(Np,1)-yk(:,2));

% INPUT COST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fu1=(u1set*ones(Nu,1) - u1(1:Nu));
fu2=(u2set*ones(Nu,1) - u2(1:Nu));


% INPUT RATE COST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fdu1=du1';    
fdu2=du2';


%THE COSTS
cost1=fy1'*Wy1*fy1     + fy2'*Wy2*fy2        ;
cost2=fu1'*Wu1*fu1     + fu2'*Wu2*fu2          ;
cost3=fdu1'*Wdu1*fdu1  + fdu2'*Wdu2*fdu2             ;

cost=cost1  +   cost2 +  cost3  ;

end



function [c,ceq] = nlfun(u,r1,r2,uo,xo,upast,ypast,Np,Nu,x1min,x2min...
    ,x1max,x2max,du1min,du2min,du1max,du2max,dy,ho,Co,P,alpha);

% yk=output_prediction(du,xo,uo,upast,Np,Nu,dy,false);
% yk=output_prediction(u,xo,uo,upast,Np,Nu,dy,ho,Co) ;
% yk=output_prediction(u,xo,uo,upast,ypast,Np,Nu,dy) ;
yk=output_prediction(u,xo,uo,upast,ypast,Np,Nu,dy,ho,Co);


% du1 = zeros(Nu,1); 
% du2 = zeros(Nu,1);  
% du3 = zeros(Nu,1);

% u1 = zeros(Nu,1); 
% u2 = zeros(Nu,1);  
% u3 = zeros(Nu,1);


% du1 = zeros(Nu,1); 
% du2 = zeros(Nu,1);  
% 
% 
% 
% u1 = zeros(Nu,1); 
% u2 = zeros(Nu,1);  


u1 = u(1:Nu);
u2 = u(Nu+1:2*Nu);



%%%%%%%%%%%%%%%%%%%%%%%%
    du1(1)=u1(1)- uo(1)  ;   du2(1)=u2(1)- uo(2) ;    
for k=2:Nu
    du1(k)= u1(k)- u1(k-1)  ;  du2(k)= u2(k)- u2(k-1) ;
end


y1k=yk(:,1)   ; %output 1
y2k=yk(:,2)    ; %output 2


%%%%%%%%%%%%%%

y1_min_colvector = x1min*ones(Np,1);
y1_max_colvector = x1max*ones(Np,1);

% du1_min_colvector = du1min*ones(Nu,1);
% du1_max_colvector = du1max*ones(Nu,1);
du1_min_colvector = du1min*ones(Nu,1);
du1_max_colvector = du1max*ones(Nu,1);

y2_min_colvector = x2min*ones(Np,1);
y2_max_colvector = x2max*ones(Np,1);
% du2_min_colvector = du2min*ones(Nu,1);
% du2_max_colvector = du2max*ones(Nu,1);
du2_min_colvector = du2min*ones(Nu,1);
du2_max_colvector = du2max*ones(Nu,1);

%The inequality constraint functions are defined
%ineq_du_min is the inequality constraint function on du_min
%ineq_du_max is the inequality constraint function on du_min
%ineq_y_min is the inequality constraint function on y_min
%ineq_y_max is the inequality constraint function on y_max

ineq_y1_min = y1_min_colvector - y1k   ;
ineq_y1_max = y1k - y1_max_colvector  ;
%disp(u1_min_colvector)
%disp(u1)
% du1_min_colvector
% du1

ineq_du1_min = du1_min_colvector - du1';
ineq_du1_max = du1' - du1_max_colvector;


ineq_y2_min = y2_min_colvector - y2k;
ineq_y2_max = y2k - y2_max_colvector;

ineq_du2_min = du2_min_colvector - du2';
ineq_du2_max = du2' - du2_max_colvector;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % LYAPUNOV CONSTRAINTS
    % u_first_move=uo ;
    % [yk,h_update,C_update] = lstm_predict(u_first_move,upast,uo, 1, 1,ho,Co)  ;
    % hk=double(h_update) ; ck=double(C_update) ;
    g=[] ;
    % rhomin=2 ;  %rhomin=2 ;372
    rhomin = alpha ;

    % rhomin=372 ;  %rhomin=2 ;372
    % P=[1060,  22 ;
    %    22,   0.52] ;   %  P=10    P=eye(20)

    dt=0.01 ;
    k=15 ;% k=0.15

    yo=xo ; y_set=[r1;r2];
    % xt=[y_set-xo] ;  %     xt=[ck;hk]% xt=[xo-y_set] 
    xt=[xo-y_set] ;  %     xt=[ck;hk]% xt=[xo-y_set]
       % xo=[Co;ho]   %      xo=[Co;ho]
    
    V=xt'*P*xt ;
    
    y_k=yk(end,:);
    x_dot=(y_k'-yo)/dt 
    V_dot=2*xt'*P*x_dot 
    if V>rhomin
        g=[g;V_dot+k*abs(V/100)]
        % g=[V_dot+k*abs(V)]
    elseif V<=rhomin
        xNN= [y_k'-y_set] 
        V_xNN=xNN'*P*xNN 
        g=[g;V_xNN-rhomin]
        % g=[V_xNN-rhomin]
    % x=[]
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c = [ineq_y1_min; ineq_y1_max; ineq_du1_min; ineq_du1_max;...
%      ineq_y2_min; ineq_y2_max; ineq_du2_min; ineq_du2_max] ;


c = [ineq_y1_min; ineq_y1_max; ineq_du1_min; ineq_du1_max;...
     ineq_y2_min; ineq_y2_max; ineq_du2_min; ineq_du2_max ; g] 
 

ceq = []   ;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xPx, Tconst]=terminal_const(u,xo,uo,upast,ypast,Np,Nu,dy,ho,Co,r1,r2,P,alpha);

yk=output_prediction(u,xo,uo,upast,ypast,Np,Nu,dy,ho,Co);

% yk=output_prediction(u,xo,uo,upast,ypast,Np,Nu,dy,ho,Co)

% [xk]=output_prediction(xo,uo,ko,du,hstep,Nsteps,Np,Nu);


xPx = [yk(Np,:)-[r1,r2]]*P*[yk(Np,:)-[r1,r2]]'   ;
Tconst=[yk(Np,:)-[r1,r2]]*P*[yk(Np,:)-[r1,r2]]' -  alpha    ;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function yk=output_prediction(u,xo,uo,upast,ypast,Np,Nu,dy);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function yk=output_prediction(u,xo,uo,upast,ypast,Np,Nu,dy,ho,Co);

% % y_k = lstm_predict_update(xo,uo,du,upast,Nu,Np,update_state) ;
% y_k = LSTM_output_prediction(xo,uo,u,upast,Nu,Np,ho,Co) ;
[y_k,~,~] = LSTM_output_prediction(xo,uo,u,upast,ypast,Nu,Np,ho,Co) ;

yk= y_k + repmat(dy',Np,1)  ;


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [y_k,h1_update,C1_update] = LSTM_output_prediction(xo,uo,uu,upast,ypast,Nu,Np,ho,Co)

n_input=2 ;
y_k = zeros(Np,n_input) ;

xo=xo(1:2)  ;


u1 = uu(1:Nu);
u2 = uu(Nu+1:2*Nu);


%%%%%%%%%%%%%%%%%%%%%%%%%
u_horizon = [u1'; u2'];

    % input to the network [u1_1;  u2_1 ; u3_1 ; u1_2 ;  u2_2 ;  u3_2 ];
    % For a three input system with prediction horizon of five
    % u=[ 1     2     3
    %     1     2     3
    %     1     2     3
    %     1     2     3
    %     1     2     3]

% Change the dimension of u

u=u_horizon' ;
uo=uo' ;
upast=upast' ;

y_present=xo' ;
y_past=ypast' ;

for i = 1:Np
        % if (i == 1)
        %     u_NN = [uo,upast] ;
        % end
        % 
        % if (i == 2)
        %     u_NN = [u(1,:),uo]  ;
        % end
        % 
        % if (i >= 3 && i<Nu)
        %     u_NN = [u(i-1,:),u(i-2,:)] ;
        % end
        % 
        % if i>=Nu
        %     u_NN = [u(Nu,:),u(Nu,:)] ;
        % end

         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (i == 1)
            u_NN = [u(1,:),uo]; % [u(1), u(0)]
        elseif (i <= Nu)
            u_NN = [u(i,:),u(i-1,:)]; % [u(2),u(1)] if i=2
        else
            u_NN = [u(Nu,:),u(Nu,:)];
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        if (i == 1)
            y_NN = [y_present,y_past];
        elseif (i == 2)
            y_NN = [y_k(1,:),y_present];
        else
            y_NN = [y_k(i-1,:),y_k(i-2,:)];
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % if (i == 1)
        %     y_NN = [y_k(1,:),y_present];
        % else
        %     y_NN = [y_k(i,:),y_k(i-1,:)];
        % end

         % Scaling the input the network      
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % u1_min = 0.5031 ; u2_min =-64.2379 ;
        % u1_max = 5.6972 ; u2_max = 5.9967e+04 ;
        % 
        % y1_min = 0.2157 ; y2_min = 311.8687  ;
        % y1_max = 3.9433 ; y2_max = 578.4861 ;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


            % u1_min =  0.5    ;  u2_min =  -1.8175e+05  ;  
            % u1_max =  7.5     ; u2_max =  4.5931e+04  ;  
            % 
            % 
            % y1_min =  1.4375  ; y2_min = 392.6911 ;
            % y1_max =  2.4971  ; y2_max = 435.2007 ;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            u1_min =  0.5    ;  u2_min =   -3.3797e+05 ;  
            u1_max =  7.5     ; u2_max =  8.3095e+04  ;  


            y1_min =   0.2157  ; y2_min = 311.8687 ;
            y1_max =  3.9433  ; y2_max = 578.4861 ;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        u1_1=scale_input2(u_NN(1),u1_min,u1_max,0,1)  ;
        u2_1=scale_input2(u_NN(2),u2_min,u2_max,0,1)  ;
        
 
        u1_2=scale_input2(u_NN(3),u1_min,u1_max,0,1)   ;
        u2_2=scale_input2(u_NN(4),u2_min,u2_max,0,1)  ;
       
        y1_1=scale_input2(y_NN(1),y1_min,y1_max,0,1);  
        y2_1=scale_input2(y_NN(2),y2_min,y2_max,0,1); 
      

        y1_2=scale_input2(y_NN(3),y1_min,y1_max,0,1); 
        y2_2=scale_input2(y_NN(4),y2_min,y2_max,0,1); 
        

        uNN_scaled=[y1_1;  y2_1;y1_2;  y2_2 ; u1_1;u2_1;u1_2;u2_2] ;


    [yk_scaled, h1_update,C1_update] = lstm_network(uNN_scaled,ho,Co) ;

    ho=h1_update ;Co=C1_update;

    y1k=scale_input2(yk_scaled(1),0,1,y1_min,y1_max)  ;
    y2k=scale_input2(yk_scaled(2),0,1,y2_min,y2_max)  ;
  
    y_k(i,:) = [y1k , y2k]  ;

 end

end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y=scale_input2(x,xmin,xmax,ymin,ymax)

y=ymin + ((x-xmin)*(ymax-ymin))/(xmax-xmin) ;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ypred, h1_update,C1_update] = lstm_network(xt,ho,Co) ;

% load('training_data_simple.mat')
load('lstm_cstr_data.mat')
% net_inputs=4;
num_hidden_units=10 ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the model weights and biases for each LSTM layer
% First LSTM Layer
Wi1 = Wu1(1:num_hidden_units,:);  % Forget gate input weights (layer 1)
Ui1 = WR1(1:num_hidden_units,:);  % Forget gate recurrent weights (layer 1)
bi1 = b1(1:num_hidden_units,:);  % Forget gate biases (layer 1)

Wf1 = Wu1(num_hidden_units+1:2*num_hidden_units,:);  % Input gate input weights (layer 1)
Uf1 = WR1(num_hidden_units+1:2*num_hidden_units,:);  % Input gate recurrent weights (layer 1)
bf1 = b1(num_hidden_units+1:2*num_hidden_units,:);  % Input gate biases (layer 1)

Wc1 = Wu1(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate input weights (layer 1)
Uc1 = WR1(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate recurrent weights (layer 1)
bc1 = b1(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate biases (layer 1)

Wo1 = Wu1(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state input weights (layer 1)
Uo1 = WR1(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state recurrent weights (layer 1)
bo1 = b1(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state biases (layer 1)



% % Second LSTM Layer
% Wi2 = Wu2(1:num_hidden_units,:);  % Forget gate input weights (layer 1)
% Ui2 = WR2(1:num_hidden_units,:);  % Forget gate recurrent weights (layer 1)
% bi2 = b2(1:num_hidden_units,:);  % Forget gate biases (layer 1)
% 
% Wf2 = Wu2(num_hidden_units+1:2*num_hidden_units,:);  % Input gate input weights (layer 1)
% Uf2 = WR2(num_hidden_units+1:2*num_hidden_units,:);  % Input gate recurrent weights (layer 1)
% bf2 = b2(num_hidden_units+1:2*num_hidden_units,:);  % Input gate biases (layer 1)
% 
% Wc2 = Wu2(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate input weights (layer 1)
% Uc2 = WR2(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate recurrent weights (layer 1)
% bc2 = b2(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate biases (layer 1)
% 
% Wo2 = Wu2(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state input weights (layer 1)
% Uo2 = WR2(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state recurrent weights (layer 1)
% bo2 = b2(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state biases (layer 1)


% Wy=poly_net.Layers(3,1).Weights ;
% by=poly_net.Layers(3,1).Bias ;

% % Initialize hidden and cell states for both LSTM layers
% h1 = zeros(num_hidden_units, 1);  % Hidden state for layer 1
% C1 = zeros(num_hidden_units, 1);  % Cell state for layer 1
% 
% h2 = zeros(num_hidden_units, 1);  % Hidden state for layer 2
% C2 = zeros(num_hidden_units, 1);  % Cell state for layer 2

% Assuming 'testData' is the input sequence you want to predict on

% testData=test_data_u;
% [num_features, num_time_steps] = size(testData);
% 
% % Initialize an array to store the outputs at each time step
% pH_pred = zeros(net_inputs, num_time_steps);
% outputSequence = zeros(num_hidden_units, num_time_steps);
    
    % ============================
    % First LSTM Layer Calculations
    % ============================
    
    % Forget gate (layer 1)
    f1 = sigmoid(Wf1 * xt + Uf1 * ho + bf1) ;
    
    % Input gate (layer 1)
    i1 = sigmoid(Wi1 * xt + Ui1 * ho + bi1);
    
    % Candidate cell state (layer 1)
    C1_candidate = tanh(Wc1 * xt + Uc1 * ho + bc1);
    
    % Update cell state (layer 1)
    C1 = f1 .* Co  + i1 .* C1_candidate;  % original
    % C1 = i1 .* C1  + f1 .* C1_candidate;
    
    % Output gate (layer 1)
    o1 = sigmoid(Wo1 * xt + Uo1 * ho + bo1);
    
    % Update hidden state (layer 1)
    h1 = o1 .* tanh(C1);
    
    % % The output of the first LSTM layer (h1) becomes the input for the second LSTM layer
    % 
    % % ============================
    % % Second LSTM Layer Calculations
    % % ============================
    % 
    % % Forget gate (layer 2)
    % f2 = sigmoid(Wf2 * h1 + Uf2 * h2 + bf2);
    % 
    % % Input gate (layer 2)
    % i2 = sigmoid(Wi2 * h1 + Ui2 * h2 + bi2);
    % 
    % % Candidate cell state (layer 2)
    % C2_candidate = tanh(Wc2 * h1 + Uc2 * h2 + bc2);
    % 
    % % Update cell state (layer 2)
    % C2 = f2 .* C2 + i2 .* C2_candidate;
    % 
    % % Output gate (layer 2)
    % o2 = sigmoid(Wo2 * h1 + Uo2 * h2 + bo2);
    % 
    % % Update hidden state (layer 2)
    % h2 = o2 .* tanh(C2)  ;

    % ypred=Wy*h2+by  ;

    ypred=Wy*h1+by  ;
    
    % Store the output of the second LSTM layer
    % outputSequence(:, t) = h2;
    % 
    % ypred=Wy*outputSequence+by 

h1_update=h1;
C1_update=C1 ; 
% h2_update=h2 ; 
% C2_update=C2 ;


% % The final output is the sequence of hidden states from the second LSTM layer
% disp('LSTM Model Output from First Principles (Two LSTM Layers):');
% disp(outputSequence);

% ============================
% Fully Connected Layer Calculation
% ============================

% % For regression, we take the output of the last time step (h2)
% finalHiddenState = h2;  % Output of the last time step from the second LSTM layer
% 
% % Apply fully connected layer
% fcOutput = Wy * finalHiddenState + by;
% 
% % ============================
% % Regression Layer
% % ============================
% % In a regression layer, the output is typically the direct prediction, 
% % so we assume 'fcOutput' is the predicted value
% 
% predictedValue = fcOutput;
% 
% % Display the final predicted value
% disp('Final Predicted Value:');
% disp(predictedValue);

% updatin cell and hidden states



end

% LSTM model prediction ends here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Sigmoid function definition
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end