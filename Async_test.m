clear;
mex_all;
% Load dataset, all normalized and transposed.

load 'rcv1.small.mat'; %% Small RCV1
 X=X';
 X = [ones(size(X,1),1) X];
 X=X';
% [y,X]=libsvmread('rcv1_test.binary');
%  X=X';

% load 'real-sim.mat'
% X = [ones(size(X,1),1) X];
% X=X';

% load 'rcv1_large.mat'; %% Large RCV1
% load 'news20_binary.mat'
% X = [ones(size(X,1),1) X];
% X=X';

% load 'Covtype.mat'
%  X = [ones(size(X,1),1) X];
%  X=X';
% X=sparse(X);

[Dim, N] = size(X);

%%% Set Params
passes =300;
model = 'logistic';  % least_square / logistic
regularizer = 'L1';
init_weight = repmat(0, Dim, 1); % Initial weight
lambda1 = 10^(-6); % L2.
L = (0.25 * max(sum(X.^2, 1)) + lambda1);
sigma = lambda1;
is_sparse = issparse(X);
Mode = 1;
is_plot = true;
fprintf('Model: %s-%s\n', regularizer, model);

%%% Thread Number
thread_no = 4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%S2VRG
algorithm = 'S2VRG';
step_size =1.8/L;
loop = int64(passes / 3); % 3 passes per loop
fprintf('Algorithm: %s\n', algorithm);
tic;
hist1 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1,...
    L, step_size, loop, is_sparse, sigma, thread_no);
time = toc;
fprintf('Time: %f seconds \n', time);
% [hist1_num, t_] = size(hist1);
X_S2VRG=zeros(1,passes/3+1);
per=4;
for i = 2:passes/3+1
    if 2^(i-2)<=2*per
        X_S2VRG(i)=X_S2VRG(i-1)+1+(2^(i-2))/per;
    else
        X_S2VRG(i)=X_S2VRG(i-1)+3;
    end
end
hist1_iter = [X_S2VRG', hist1(1:2:end)];
hist1_time = [hist1(2:2:end), hist1(1:2:end)];

%%% KroMagnon
algorithm = 'KroMagnon';
 step_size =1/L;
loop = int64(passes / 3); % 3 passes per loop
fprintf('Algorithm: %s\n', algorithm);
tic;
hist3 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1,...
    L, step_size, loop, is_sparse, sigma, thread_no);
time = toc;
fprintf('Time: %f seconds \n', time);
X_SVRG = [0:3:passes]';
hist3_iter = [X_SVRG, hist3(1:2:end)];
hist3_time = [hist3(2:2:end)-hist3(2), hist3(1:2:end)];

%%% ASAGA
algorithm = 'ASAGA';
step_size =0.1/L;
loop = int64((passes - 1) * N); % One Extra Pass for initialize SAGA gradient table.
fprintf('Algorithm: %s\n', algorithm);
tic;
hist4 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1,...
    L, step_size, loop, is_sparse, sigma, thread_no);
time = toc;
fprintf('Time: %f seconds \n', time);
X_SAGA = [0 1 2:3:passes - 2]';
hist4_iter = [X_SAGA, hist4(1:2:end)];
hist4_time = [hist4(2:2:end)-hist4(2), hist4(1:2:end)];

%%% Plot
if(is_plot)
    %optimal = 0.013298006616553; % For L2-logistic on RCV1-train, lambda = 10^(-7)
    optimal = min([min([hist1_iter(:, 2)]), min([hist3_iter(:, 2)]),min([hist4_iter(:, 2)])]);%min([hist2_iter(:, 2)]), 
    minval = optimal - 2e-16;
    aa = max(max([hist1_iter(:, 2)])) - minval;
    b = 2;

    figure(101);
    set(gcf,'position',[200,100,386,269]);
    semilogy(hist1_iter(1:b:end,1), abs(hist1_iter(1:b:end,2) - minval),'g--x','linewidth',1.2,'markersize',4.5);
%     hold on,semilogy(hist2_iter(1:b:end,1), abs(hist2_iter(1:b:end,2) - minval),'m--d','linewidth',1.2,'markersize',4.5);
    hold on,semilogy(hist3_iter(1:b:end,1), abs(hist3_iter(1:b:end,2) - minval),'b--+','linewidth',1.2,'markersize',4.5);
    hold on,semilogy(hist4_iter(1:b:end,1), abs(hist4_iter(1:b:end,2) - minval),'k-.o','linewidth',1.2,'markersize',4.5);
    hold off;
    xlabel('Number of effective passes');
    ylabel('Objective minus best');
            axis([0 700, 1E-16, aa]);
    legend('S2VRG', 'KroMagnon', 'ASAGA');% 'VRSGD',
    
    figure(1);
    semilogy(hist1_time(1:b:end,1), abs(hist1_time(1:b:end,2) - minval),'g--x','linewidth',1.2,'markersize',4.5);
%     hold on,semilogy(hist2_time(1:b:end,1), abs(hist2_time(1:b:end,2) - minval),'m--d','linewidth',1.2,'markersize',4.5);
    hold on,semilogy(hist3_time(1:b:end,1), abs(hist3_time(1:b:end,2) - minval),'b--+','linewidth',1.2,'markersize',4.5);
    hold on,semilogy(hist4_time(1:b:end,1), abs(hist4_time(1:b:end,2) - minval),'k-.o','linewidth',1.2,'markersize',4.5);
    hold off;
    xlabel('Cost of Time ');
    ylabel('Objective minus best');
               axis([0 35, 1E-16, aa]);
   legend('S2VRG', 'KroMagnon', 'ASAGA');% 'VRSGD',
end
