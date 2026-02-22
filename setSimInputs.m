function simIn = setSimInputs(simIn, load_P, load_P1, load_P2, load_P3)
%SETSIMINPUTS Sets the inputs for Simulink simulation
%   Sets the inputs for Simulink simulation
simIn = setVariable(simIn, "load_P", load_P);
simIn = setVariable(simIn, "load_P1", load_P1);
simIn = setVariable(simIn, "load_P2", load_P2);
simIn = setVariable(simIn, "load_P3", load_P3);
end

