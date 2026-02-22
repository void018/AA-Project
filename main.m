%%% default values
defaults = [
    "generator_Vrms", 13800 ...
    "load_P", 40e6 ...
    "load_QL", 15e6 ...
];
start_time = datetime;
load_P = 40e6;
load_P1 = load_P * 0.3;
load_P2 = load_P * 0.3;
load_P3 = load_P * 0.3;

%%% initialization
%load_P_values = [40e6, 40e3, 40, 0];
n = 6;
load_P_values = logspace(0, 5, n);
load_P1_values = logspace(0, 4.5, n);
load_P2_values = logspace(0, 4.5, n);
load_P3_values = logspace(0, 4.5, n);

%load_P_values = [9e9];
%load_Q_values = [2e9];

num_of_runs = length(load_P_values) * length(load_P1_values) * length(load_P2_values) * length(load_P3_values);

T = array2table(zeros(num_of_runs, 20), 'VariableNames', ["No", "negative_sequence", "V_A", "V_B", "V_C", "I_A", "I_B", "I_C", "V_a", "V_b", "V_c", "I_a", "I_b", "I_c", "phase_B", "phase_C", "phase_a", "phase_b", "phase_c", "faulted"]);


simIn = Simulink.SimulationInput("open_phase_model");
clear simIns;
counter = 1;
for cur_P=load_P_values
    for cur_P1=load_P1_values
        for cur_P2=load_P2_values
            for cur_P3=load_P3_values
                simIn = setSimInputs(simIn, cur_P, cur_P1, cur_P2, cur_P3);
                simIns(counter) = simIn;
                counter = counter + 1;
            end
        end
    end
end

% perform the simulation
%outs = sim(simIns, "UseFastRestart","on");
outs = parsim(simIns);

% reset counter
counter = 1;
for out=outs
    % format simulation output
    logsout = out.logsout;

    negative_sequence = getVarValue(logsout,"negative_sequence");
    V_A = round(getVarValue(logsout,"V_A"));
    V_B = round(getVarValue(logsout,"V_B"));
    V_C = round(getVarValue(logsout,"V_C"));
    I_A = round(getVarValue(logsout,"I_A"), 1);
    I_B = round(getVarValue(logsout,"I_B"), 1);
    I_C = round(getVarValue(logsout,"I_C"), 1);
    V_a = round(getVarValue(logsout,"V_a"));
    V_b = round(getVarValue(logsout,"V_b"));
    V_c = round(getVarValue(logsout,"V_c"));
    I_a = round(getVarValue(logsout,"I_a"), 1);
    I_b = round(getVarValue(logsout,"I_b"), 1);
    I_c = round(getVarValue(logsout,"I_c"), 1);
    phase_B = round(getVarValue(logsout,"phase_B"), 2);
    phase_C = round(getVarValue(logsout,"phase_C"), 2);
    phase_a = round(getVarValue(logsout,"phase_a"), 2);
    phase_b = round(getVarValue(logsout,"phase_b"), 2);
    phase_c = round(getVarValue(logsout,"phase_c"), 2);

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
