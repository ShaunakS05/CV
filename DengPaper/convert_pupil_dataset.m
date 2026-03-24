load('Pupil_dataset.mat');  % loads Pupil_data

n = numel(Pupil_data);
Pupil_data_py = struct([]);

for i = 1:n
    Pupil_data_py(i).Subject = Pupil_data(i).Subject;
    Pupil_data_py(i).Age = Pupil_data(i).Age;
    Pupil_data_py(i).Group = Pupil_data(i).Group;
    Pupil_data_py(i).Wisc = Pupil_data(i).Wisc;

    te = Pupil_data(i).Task_epocs;
    if istable(te)
        S = struct();
        vars = te.Properties.VariableNames;
        for k = 1:numel(vars)
            S.(vars{k}) = te.(vars{k});
        end
        Pupil_data_py(i).Task_epocs = S;
    else
        Pupil_data_py(i).Task_epocs = te;
    end

    Pupil_data_py(i).Task_data = [];
end

save('Pupil_dataset_py.mat', 'Pupil_data_py', '-v7');