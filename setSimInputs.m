function simIn = setSimInputs(simIn, load_P, load_QL)
%SETSIMINPUTS Sets the inputs for Simulink simulation
%   Sets the inputs for Simulink simulation
simIn = setVariable(simIn, "load_P", load_P);
simIn = setVariable(simIn, "load_QL", load_QL);
end

