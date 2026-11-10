function [sys,x0,str,ts] = CSTR_PLANT_sfun(t,x,u,flag,x0)

%[sys,x0,str,ts] = smt_sfun(t,x,u,flag,x0)
    %Input arguments
    %t = time variable.
    %x = State variables.
    %u = input variables.
    %flag = indicates which group of information is requested by simulink.

    %Returned Parameters
    %sys = vector of results requested by simulink. type of results requested
    %depends on the value of flag.
    %x0 = vector of initial conditions.
    %str = [].
    %ts = sampling time and time offsets.

    % Initial state

    switch flag
        
        case 0
            
            %Initialization
            %sys is required to contain the following
            %(i)Number of continuous states. (ii) Number of discrete states.
            %(iii) Number of outputs. (iv) Number of inputs.
            %(v) Information on whether direct algebraic io feedthrough exists.
            %(vi) Sample time (1 for continuous sampling).
            
            data_struct = simsizes;
            data_struct.NumContStates = 2;
            data_struct.NumOutputs = 2;
            data_struct.NumInputs = 2;
            data_struct.DirFeedthrough = 0;
            data_struct.NumSampleTimes = 1;
            data_struct.NumDiscStates = 0;
            
            sys = simsizes(data_struct);
            str = [];
            ts = [0,0];
            % x0 = [1.9537; 401.8727]+[-1; 63.6];
            x0 = [0.95; 448];            
            % x0 = [0.5; 445];
            % x0 = [1.9537; 401.8727];
            
        case 1
            %States derivative calculations
            sys = ode_set(x, u); 
        case 3
            %Output calculations
            sys = x;
            
        case {2,4,9}
            
            %flag = 2, Updating of discrete equations (Not performed).
            %flag = 4, Time of next variable hit. (Not performed).
            %flag = 9, Calculations at the end of simulation run. (Not performed).
            
            sys = [];
            
        otherwise
            
            error(['unhandled flag = ',num2str(flag)]);
    end
      
end