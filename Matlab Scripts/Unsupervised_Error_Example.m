% Read in agreement counts 
agreement_rates = readtable('Agreement Counts Example.xlsx');

% table to an array
agreement_rates = table2array(agreement_rates);

% set upper and lower bounds a choose an x0 (multistart is going to be
% used, so this choice really doesn't matter)
lb = zeros(57,1);
ub = ones(57,1);
x0 = lb;

% find a solution for c1; multistart a couple thousand times to avoid local
% minima
rng default % For reproducibility
options = optimoptions(@fmincon,'Algorithm','sqp','MaxFunctionEvaluations',100000,'MaxIterations',1000);
problem = createOptimProblem('fmincon','objective',...
    @(x)c1(x),'x0',x0,'lb',lb,'ub',ub,'options',options,'nonlcon',@(x)Unsupervised_Error_Constraints_Multi_Class(x,agreement_rates,36));
ms = MultiStart;
ms = MultiStart(ms,'UseParallel',true);
[opt_c1,opt_cost_c1] = run(ms,problem,3000);

% The error rates are
e1_c1 = opt_c1(1);  % Modified Yarowsky
e2_c1 = opt_c1(2);  % Label Propagation
e3_c1 = opt_c1(3);  % COP-KMEANS
e4_c1 = opt_c1(4);  % S4VM
e5_c1 = opt_c1(5);  % Updated


errors = opt_c1(1:5);
errors = array2table(errors)
filename = 'Unsupervised_Error_Example.xlsx'
writetable(errors,filename)
