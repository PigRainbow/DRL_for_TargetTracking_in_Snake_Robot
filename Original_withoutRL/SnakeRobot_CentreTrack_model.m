function [xDot] = SnakeRobot_CentreTrack_model(t,x,config)
%% Initial parameters
% robot parameters
ct = config.ct;  % Tangential friction coefficient
cn = config.cn;  % Normal friction coefficient
N  = config.N;  % Number of links
m  = config.m;  % Mass of each link
l  = config.l;  % Half Length of each link
J  = 1/3 * m * l^2;  % Moment Inertia of each link

% reference movment define
ampli_u = config.ampli_u;
omega_u = config.omega_u;
delta_u = config.delta_u;

% controler parameters
Kp = config.Kp;
Kd = config.Kd;

% State vector define
theta = x(1:N); 
Px = x(N+1);  % CM of the robot on X-axis  
Py = x(N+2);  % CM of the robot on Y-axis
thetaDot = x(N+3:2*N+2);
PxDot = x(2*N+3);
PyDot = x(2*N+4);

%% Auxiliary matrices definition
B = diag(ones(N-1,1),1);
A = eye(N-1,N) + B(1:N-1,:);
D = eye(N-1,N) - B(1:N-1,:);
e = ones(N,1);
E = [e, zeros(N,1); zeros(N,1), e];
Stheta = diag(sin(theta));
Ctheta = diag(cos(theta));
%tau_R = -diag(cn*ones(N,1)) * diag(J*ones(N,1)) * thetaDot; % new

% Equation 2.9
T = [D; 1/N * e'];

% Equation 2.34a, 2.34b, 2.34c, 2.34d
V = A' * (D * D')^(-1) * A;
K = A' * (D * D')^(-1) * D;
M = J * eye(N) + m * l^2 * Stheta * V * Stheta + m * l^2 * Ctheta * V * Ctheta;
W = m * l^2 * Stheta * V * Ctheta - m * l^2 * Ctheta * V * Stheta;

% Equation 2.11a, 2.11b 
% X = -l * K' * cos(theta) + e * Px;  % Position of the CM of each link on Global X-axis
% Y = -l * K' * sin(theta) + e * Py;  % Position of the CM of each link on Global Y-axis
X = T^(-1) * [-l * A * cos(theta); Px];
Y = T^(-1) * [-l * A * sin(theta); Py];

phi = D * theta;
phiDot = D * thetaDot;

% Initialize previous orientation
u = zeros(N-1,1);
phi_ref = zeros(N-1,1);
phi_refDot = zeros(N-1,1);
theta_ref = zeros(N,1);
theta_refDot = zeros(N,1);

% PD controller
for i = 1:N
    theta_ref(i) = ampli_u * sin(omega_u * t + (i-1) * delta_u);
    theta_refDot(i) = ampli_u * omega_u * cos(omega_u * t + (i-1) * delta_u);
end

for j = 1:N-1
    phi_ref(j) = theta_ref(j) - theta_ref(j+1);
    phi_refDot(j) = theta_refDot(j) - theta_refDot(j+1);
    %u(j) = Kp * (phi_ref(j) - phi(j)) - Kd * (phiDot(j));
    u(j) = Kp * (phi_ref(j) - phi(j)) + Kd * (phi_refDot(j) - phiDot(j));
end


% Equation 2.12a and 2.12b
XDot =  l * K' * Stheta * thetaDot + e * PxDot;
YDot = -l * K' * Ctheta * thetaDot + e * PyDot;

% Equation 2.25
fR = -[ct * Ctheta^2 + cn * Stheta^2, (ct - cn) * Stheta * Ctheta; ...
       (ct - cn) * Stheta * Ctheta, ct * Stheta^2 + cn* Ctheta^2] * [XDot; YDot];


thetaDot_2 = M^(-1) * (D' * u - W * diag(thetaDot) * thetaDot + l * Stheta * K * fR(1:N) - l * Ctheta * K * fR(N+1:2*N));
%thetaDot_2 = M^(-1) * (D' * u - W * diag(thetaDot) * thetaDot + l * Stheta * K * fR(1:N) - l * Ctheta * K * fR(N+1:2*N));
PxDot_2 = e' * fR(1:N) / (N * m);
PyDot_2 = e' * fR(N+1:2*N) / (N * m);

% State vector
xDot=[thetaDot; PxDot; PyDot; thetaDot_2; PxDot_2; PyDot_2];
