% Set robot parameters
config.N = 5; 
config.l = 0.14; 
config.m = 1; 
config.ct = 0.5;  
config.cn = 3; 
config.Kp = 100;
config.Kd = 1;


% Create predefined environment interface for the snake robot
agent = [];
env = SnakeRobot_env_DDPG(config, agent);

% Obtain the observation and action specification from the environment interface
observationInfo = getObservationInfo(env);
actionInfo = getActionInfo(env);

% Define network architecture
num_Observations = observationInfo.Dimension(1);
num_Actions = actionInfo.Dimension(1);


%% Actor network
% Scale factor defined for scaled the action to the required range
scalefactor = [4*pi/9 * ones(5, 1); 3 * ones(5, 1)];

actorNetwork = [
    featureInputLayer(num_Observations, Name="ActorIn")
    fullyConnectedLayer(256, Name="ActorFC1")
    reluLayer
    fullyConnectedLayer(128, Name="ActorFC2")
    reluLayer
    fullyConnectedLayer(num_Actions, Name="ActorFC3")
    tanhLayer
    scalingLayer(Scale=scalefactor,Bias=0, Name="ActorOut")];  

actor = rlContinuousDeterministicActor(actorNetwork, observationInfo, actionInfo);


%% Critic network
statePath = [
    featureInputLayer(num_Observations, Name="StateIn")
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(128, Name="StateOut")];
    
actionPath = [
    featureInputLayer(num_Actions, Name="ActionIn")
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(128, Name="ActionOut")];

commonPath = [
    additionLayer(2, Name="Add")
    reluLayer
    fullyConnectedLayer(1, Name="CriticOutput")];

criticNetwork = dlnetwork;
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,"StateOut","Add/in1");
criticNetwork = connectLayers(criticNetwork,"ActionOut","Add/in2");
critic = rlQValueFunction(criticNetwork, observationInfo, actionInfo);

%% Create DDPG agent
actorOpts = rlOptimizerOptions(...
    Algorithm='adam', ...        
    LearnRate=1e-5, ...  
    GradientThreshold=1); 

criticOpts = rlOptimizerOptions(...
    Algorithm='adam', ...        
    LearnRate=1e-4, ...  
    GradientThreshold=1, ...       
    L2RegularizationFactor=1e-3); 

agentOptions = rlDDPGAgentOptions(...
    SampleTime=0.04,...
    ActorOptimizerOptions=actorOpts,...
    CriticOptimizerOptions=criticOpts,...
    TargetSmoothFactor=1e-3,...
    ExperienceBufferLength=1e6,...
    DiscountFactor=0.99,...
    MiniBatchSize=256);

% Front five are u_theta_ref，rest of actions are u_theta_refDot 5% to 0.1%
agentOptions.NoiseOptions.StandardDeviation = [0.7 * ones(5, 1); 1.5 * ones(5, 1)];
agentOptions.NoiseOptions.StandardDeviationDecayRate = 5.3e-6;
agent = rlDDPGAgent(actor, critic, agentOptions);

% Attach agent to the environment
env.agent = agent;

% Training options
trainOpts = rlTrainingOptions(...
    MaxEpisodes=3000,...
    MaxStepsPerEpisode=250,...
    StopTrainingCriteria="AverageReward",...
    ScoreAveragingWindowLength=30,...
    Verbose=true,...
    Plots="training-progress");
%  StopTrainingValue=170,...

% Training the agent
doTraining = true;
if doTraining
    trainingStats = train(agent, env, trainOpts);
end

% Save the trained agent
save('Trained_DDPG_agent.mat', 'agent');

% Plot the reward figure
figure;
plot(trainingStats.EpisodeIndex, trainingStats.EpisodeReward, 'Color', '#0072BD', 'LineWidth', 1); 
hold on;
plot(trainingStats.EpisodeIndex, trainingStats.AverageReward, 'Color', "#D95319", 'LineWidth', 2);  
hold off;  

%title("Episode Reward and Average Reward");
xlabel("Number of Episode");
ylabel("Reward");
legend({'Episode Reward', 'Average Reward'}, 'Location', 'best');  
grid on; 

