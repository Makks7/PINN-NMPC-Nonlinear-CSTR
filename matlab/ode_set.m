 function dxdt = ode_set(x,u)   
      CA = x(1);
      T = x(2);

      CA0 = u(1);
      Q = u(2);
      
        T0 = 300; % K
        F = 5; % m^3/hr
        V = 1; % m^3
        E = 5e4; % kJ/kmol
        k0 = 8.46e6; % m^3/kmol/hr
        dH = -1.15e4; % kJ/kmol
        Cp = 0.231; % kJ/kg/K
        R = 8.314; % kJ/kmol/K
        rhoL = 1000; % kg/m^3
        % CA0 = 4; % kmol/m^3
        % Qs = 0.0; % kJ/hr
        % Ts = 438; % K
   
      dx_1dt = (F/V) * (CA0 - CA) -  k0 * exp(-E / (R * T))* CA^2;
      dx_2dt = (F/V) * (T0 - T) + (-dH / (rhoL * Cp)) * k0 * exp(-E / (R * T))* CA^2 + ...
                  Q / (rhoL * Cp * V);
      
     
      dxdt = [dx_1dt; dx_2dt];
     
 end