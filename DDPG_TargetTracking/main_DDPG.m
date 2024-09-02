clc;
clear;

% Load the trained DDPG agent
load('Trained_DDPG_agent.mat', 'agent'); 

%% Snake robot parameter setting
config.N = 5;  % Number of links
config.l = 0.14;  % Half length of each link
config.m = 1;  % Mass of each link

% Friction coefficient
config.ct = 0.5;  % Along link x-axis
config.cn = 3;  % Along link y-axis

% PID Controller parameter
config.Kp = 100;
config.Kd = 1; 

% Time setting
tspan = 0:0.04:10;

% Reference signal
phi_r = zeros(1,config.N-1);
 
% Initial link and joint angles and velocities
theta0 = zeros(config.N,1);
thetaDot0 = zeros(config.N,1);

% Mass centers position:
Px0 = 0;
Py0 = 0;
PxDot0 = 0;
PyDot0 = 0;

% Rearrangement of boundary conditions (a - actuated, u - unactuated):
x0 = [theta0; Px0; Py0; thetaDot0; PxDot0; PyDot0];

% Initialize storage for simulation results
X1 = zeros(length(tspan), length(x0));
T1 = tspan;
X1(1, :) = x0';

for i = 2:length(tspan)
    % Get the current state
    currentState = X1(i-1, :)';

    % Use the trained agent to get the action (control input)
    action = getAction(agent, currentState);
    if iscell(action)
        action = cell2mat(action);  % Convert to numeric array if it's a cell array
    end

    u_theta_ref = action(1:config.N);
    u_theta_refDot = action(config.N+1:end);

    % Simulate the system for one time step using the action
    [~, X_temp] = ode45(@(t, x) SnakeRobot_CentreTrack_DDPG_Dynamics(t, x, config, u_theta_ref, u_theta_refDot), [tspan(i-1) tspan(i)], currentState);
    
    % Store the new state
    X1(i, :) = X_temp(end, :);
end

% Extract the center of mass position of the snake robot
Px = X1(:, config.N+1);  % Centre of mass on x-axis
Py = X1(:, config.N+2);  % Centre of mass on y-axis

% Extract the center of mass velocity of the snake robot
PxDot = X1(:, 2*config.N+3);  
PyDot = X1(:, 2*config.N+4); 

% Extract qa (phi) from X1
all_theta = rad2deg(X1(:, 1:config.N));

% Plot the result
figure(1);
set(gcf, 'Color', 'w');
plot(Px, Py, 'b', 'DisplayName', 'Trajectory', 'LineWidth', 1.5);
hold on;
plot(1, 0.3, 'r.', 'DisplayName', 'Target', 'MarkerSize', 25);
xlabel('x (m)');
ylabel('y (m)');
%title('Center of Mass Trajectory');
legend;
box on;
grid off;
xlim([-0.3, 1.2])
ylim([-0.3, 0.5]);


figure(2);
set(gcf, 'Color', 'w');
plot(T1, PxDot, 'DisplayName', '$\dot{p}_x$', 'LineWidth', 1.5);
hold on;
plot(T1, PyDot, 'DisplayName', '$\dot{p}_y$', 'LineWidth', 1.5);
xlabel('Time (s)', 'Interpreter', 'latex');
ylabel('Velocity (m/s)', 'Interpreter', 'latex');
%ylabel('Velocity (cm/s)', 'Interpreter', 'latex');
%title('Velocity $\dot{p}_x$ and $\dot{p}_y$ vs Time', 'Interpreter', 'latex');
legend({'$\dot{p}_x$', '$\dot{p}_y$'}, 'Interpreter', 'latex');
grid off;


figure(3);
set(gcf, 'Color', 'w');
hold on;
for i = 1:(config.N)
    plot(T1, all_theta(:, i), 'DisplayName', ['\theta_', num2str(i)], 'LineWidth', 1.5);
end
xlabel('Time (s)');
ylabel('Theta (degrees)');
title('Link Angles (\theta) vs Time');
legend show;
box on;
grid off;
hold off;


figure(4);
set(gcf, 'Color', 'w');
phi = zeros(length(tspan), config.N-1);
for i = 1:(config.N-1)
    phi(:,i) = all_theta(:,i) - all_theta(:,i+1);
end
hold on;
for j = 1:(config.N-1)
     plot(T1, phi(:, j), 'DisplayName', ['\phi_', num2str(j)], 'LineWidth', 1.5);
end
xlabel('Time (s)');
ylabel('\phi_ (degree)');
%title('Joint Angle (\phi) vs Time');
legend show;
box on;
grid off;
hold off;


% Calculate phi_ref
theta_ref = zeros(length(tspan), config.N);
phi_ref = zeros(length(tspan), config.N-1);
for i = 1:length(tspan)
    t = tspan(i);
    for j = 1:(config.N)
        theta_ref(i, j) = config.ampli_u * sin(config.omega_u * t + (j-1) * config.delta_u);
    end
end
for i = 1:length(tspan)
    for j = 1:(config.N-1)
        phi_ref(i, j) = theta_ref(i,j) - theta_ref(i,j+1);
    end
end


% Plot phi_ref
figure(5);
set(gcf, 'Color', 'w');
hold on;
for j = 1:config.N-1
    plot(tspan, rad2deg(phi_ref(:, j)), 'DisplayName', ['\phi_{ref,' num2str(j) '}'], 'LineWidth', 1.5);
end
xlabel('Time (s)');
ylabel('\phi_{ref} (degree)');
%title('Reference Joint Angles (\phi_{ref}) vs Time');
legend show;
box on;
grid off;
hold off;


figure(6);
set(gcf, 'Color', 'w');
e_phi = zeros(length(tspan), config.N-1);
for i = 1:(config.N-1)
    e_phi(:,i) = rad2deg(phi_ref(:,i)) - phi(:,i);
end
hold on;
for j = 1:config.N-1
    plot(T1, e_phi(:,j), 'DisplayName', ['e_{\phi_{' num2str(j) '}}'], 'LineWidth', 1.5);
end
xlabel('Time (s)');
ylabel('Joint tracking error (degree)');
%title('Joint tracking error for each joint(e_\phi) vs Time');
legend show;
box on;
grid off;
hold off;


figure(7)
set(gcf, 'Color', 'w');
B = diag(ones(config.N-1,1),1);
A = eye(config.N-1,config.N) + B(1:config.N-1,:);
D = eye(config.N-1,config.N) - B(1:config.N-1,:);
e = ones(config.N,1);
K = A' * (D * D')^(-1) * D;
X_link = zeros(length(tspan), config.N);
Y_link = zeros(length(tspan), config.N);
for i = 1:length(tspan)
    X_link(i, :) = - config.l * K' * cos(deg2rad(all_theta(i,:)')) + e * Px(i,:);
    Y_link(i, :) = - config.l * K' * sin(deg2rad(all_theta(i,:)')) + e * Py(i,:);
end

hold on;
time_index = [1, find(tspan == 10), length(tspan)];

for i = 1:config.N
    plot(X_link(time_index, i), Y_link(time_index, i), 'o', 'DisplayName', ['Link ', num2str(i)], 'LineWidth', 1.5);
end

% Plot smooth dashed lines connecting the centres of mass
for t_idx = time_index
    x_data = X_link(t_idx, :);
    y_data = Y_link(t_idx, :);

    tspan_interp = linspace(1, config.N, 100);
    x_smooth = spline(1:config.N, x_data, tspan_interp);
    y_smooth = spline(1:config.N, y_data, tspan_interp);
    plot(x_smooth, y_smooth, '--k', 'DisplayName', [num2str((t_idx-1) / 100), 's']);
end

xlabel('x (m)');
ylabel('y (m)');
title('Link Trajectories at Specific Times');
legend;
box on;
grid off;
hold off;
xlim([-1, 4]);
ylim([-1, 2]);
