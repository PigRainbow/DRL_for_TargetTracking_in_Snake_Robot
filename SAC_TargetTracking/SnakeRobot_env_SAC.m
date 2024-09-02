classdef SnakeRobot_env_SAC < rl.env.MATLABEnvironment
    properties
        config  % Environment parameters  
        agent
        state
        StepCnt 
    end

    properties(Access = protected)
    % Initialize internal flag to indicate episode termination
    IsDone = false        
    end

    methods
        function this = SnakeRobot_env_SAC(config, agent)
            % Create observation and action specification
            ObservationInfo = rlNumericSpec([2*config.N + 4 1]);
            ActionInfo = rlNumericSpec([2*(config.N) 1]);  % Direct action on theta and thetaDot
            
            % Call superclass constructor
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            
            % Set environment parameters
            this.config = config;
            this.agent = agent;
            this.state = zeros(2*this.config.N + 4, 1);  % Initialize state
            this.StepCnt = 0;  % Initialize step count
        end
        
        function InitialObservation = reset(this)
            % Reset environment
            InitialObservation = zeros(2*this.config.N+4, 1);
            this.state = InitialObservation;
            this.StepCnt = 0;
        end
        
        function [NextObservation, Reward, IsDone] = step(this,~)
            % Use SAC agent to get corrective input for theta and thetaDot
            action = getAction(this.agent, this.state);
            if iscell(action)
                action = cell2mat(action);  % Ensure it is array
            end
            
            u_theta_ref = action(1:this.config.N);
            u_theta_refDot = action(this.config.N+1:2*this.config.N);

            % Limit the action range
            if any(u_theta_ref < -4*pi/9| u_theta_ref > 4*pi/9)
                u_theta_ref = max(min(u_theta_ref, 4*pi/9), -4*pi/9);
            end

            if any(u_theta_refDot < -3 | u_theta_refDot > 3)
                u_theta_refDot = max(min(u_theta_refDot, 3), -3);
            end

            % Update state using ode45
            [~, x] = ode45(@(t, x) SnakeRobot_CentreTrack_SAC_Dynamics(t, x, this.config, u_theta_ref, u_theta_refDot), [0, 0.04], this.state);
            NextObservation = x(end, :)';
            this.state = NextObservation;

            % Increment step count
            this.StepCnt = this.StepCnt + 1;
            
            % Get reward
            Reward = computeReward(this, NextObservation);

            % Check termination
            Px = NextObservation(this.config.N+1);
            Py = NextObservation(this.config.N+2);
            PxDot = NextObservation(2*this.config.N+3);
            PyDot = NextObservation(2*this.config.N+4);
            x_target = 1;
            y_target = 0.3;
            delta_x = x_target - Px;
            delta_y = y_target - Py;
            tot_distance = sqrt(abs(delta_x)^2 + abs(delta_y)^2);
            tot_velocity = sqrt(PxDot^2 + PyDot^2);
            distance_threshold = 0.01;
            velocity_threshold = 0.01;
            IsDone = tot_distance <= distance_threshold && tot_velocity <= velocity_threshold;
            %IsDone = false;

        end

        function Reward = computeReward(this, Observation)
            % Extract theta, phi, position, velocity of CM of the whole robot
            theta = Observation(1:this.config.N);
            phi = zeros(this.config.N-1);
            for i = 1:(this.config.N-1)
                 phi(i) = theta(i) - theta(i+1);
            end
            Px = Observation(this.config.N+1);
            Py = Observation(this.config.N+2);
            PxDot = Observation(2*this.config.N+3);
            PyDot = Observation(2*this.config.N+4);

            % Reward function: To track a target
            x_target = 1;
            y_target = 0.3;
            origin_distance = sqrt(x_target^2 + y_target^2);
            delta_x = x_target - Px;
            delta_y = y_target - Py;
            tot_distance = sqrt(abs(delta_x)^2 + abs(delta_y)^2);
            tot_velocity = sqrt(PxDot^2 + PyDot^2);
            distance_threshold = 0.01;
            velocity_threshold = 0.01;
            maxstep = 250;
            
            Reward = 1 - tot_distance/origin_distance;
            if tot_distance <= distance_threshold
                % High reward for reaching the target and stopping
                if tot_velocity < velocity_threshold
                    Reward = Reward + 150 * (maxstep - this.StepCnt)/maxstep;
                else
                    Reward = Reward - 0.02 *(tot_velocity - velocity_threshold)/velocity_threshold;
                end
            end
            Reward = max(-0.25, Reward); 

            % Scale the Reward in SAC
            Reward = 1/1 * Reward;
        end
    end
end