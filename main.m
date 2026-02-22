%%% default values
defaults = [
    "generator_Vrms", 13800 ...
    "load_P", 40e6 ...
    "load_QL", 15e6 ...
];
start_time = datetime;
load_P = 40e6;
load_QL = 15e6;

%%% initialization
%load_P_values = [40e6, 40e3, 40, 0];
%load_P_values = linspace(0, 40e6, 2);
load_P_values = [80e6];

load_Q_values = [15e6];
num_of_runs = length(load_P_values) * length(load_Q_values);

T = array2table(zeros(num_of_runs, 20), 'VariableNames', ["No", "negative_sequence", "V_A", "V_B", "V_C", "I_A", "I_B", "I_C", "V_a", "V_b", "V_c", "I_a", "I_b", "I_c", "phase_B", "phase_C", "phase_a", "phase_b", "phase_c", "faulted"]);


simIn = Simulink.SimulationInput("open_phase_model");
counter = 1;
for cur_P=load_P_values
    for cur_Q=load_Q_values
        simIn = setSimInputs(simIn, cur_P, cur_Q);

        simIns(counter) = simIn;
        counter = counter + 1;
    end
end
% perform the simulation
%outs = sim(simIns, "UseFastRestart","on");
outs = sim(simIns);

% reset counter
counter = 1;
for out=outs
    % format simulation output
    logsout = out.logsout;

    negative_sequence = getVarValue(logsout,"negative_sequence");
    V_A = getVarValue(logsout,"V_A");
    V_B = getVarValue(logsout,"V_B");
    V_C = getVarValue(logsout,"V_C");
    I_A = getVarValue(logsout,"I_A");
    I_B = getVarValue(logsout,"I_B");
    I_C = getVarValue(logsout,"I_C");
    V_a = getVarValue(logsout,"V_a");
    V_b = getVarValue(logsout,"V_b");
    V_c = getVarValue(logsout,"V_c");
    I_a = getVarValue(logsout,"I_a");
    I_b = getVarValue(logsout,"I_b");
    I_c = getVarValue(logsout,"I_c");
    phase_B = getVarValue(logsout,"phase_B");
    phase_C = getVarValue(logsout,"phase_C");
    phase_a = getVarValue(logsout,"phase_a");
    phase_b = getVarValue(logsout,"phase_b");
    phase_c = getVarValue(logsout,"phase_c");

    % add data for both: healthy run & faulted run
    for j=[1, 2]
        new_data = {counter, negative_sequence(j), V_A(j), V_B(j), V_C(j), I_A(j), I_B(j), I_C(j), V_a(j), V_b(j), V_c(j), I_a(j), I_b(j), I_c(j), phase_B(j), phase_C(j), phase_a(j), phase_b(j), phase_c(j), j-1};
        T(counter, :) = new_data;
        counter = counter + 1;
    end
end

disp(T)
fprintf("simulation time: %s\n", datetime - start_time);
%out.logsout{1}.Values.Data 

% TODO: does transformer explode in Simulink?
% TODO: find reasonable P and Q values
% TODO: should we include P and Q in model training
% TODO: should we do something with power factor?
