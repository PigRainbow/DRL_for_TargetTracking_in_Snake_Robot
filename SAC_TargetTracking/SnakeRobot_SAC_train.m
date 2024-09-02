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
env = SnakeRobot_env_SAC(config, agent);

% Obtain the observation and action specification from the environment interface
observationInfo = getObservationInfo(env);
actionInfo = getActionInfo(env);


%% Actor network
% Setting the action limit range
actInfo.LowerLimit = [-4*pi/9 * ones(5, 1); -3 * ones(5, 1)];
actInfo.UpperLimit = [4*pi/9 * ones(5, 1); 3 * ones(5, 1)];
% Common input path
commonPath = [
    featureInputLayer(prod(observationInfo.Dimension), Name="netObsIn")
    fullyConnectedLayer(128)
    reluLayer(Name="commOut")];

% Path for mean value
meanPath = [
    fullyConnectedLayer(64, Name="meanIn")
    reluLayer
    fullyConnectedLayer(prod(actionInfo.Dimension), Name="meanOut")];

% Path for standard deviation
stdPath = [
    fullyConnectedLayer(64, Name="stdIn")
    reluLayer
    fullyConnectedLayer(prod(actionInfo.Dimension))
    softplusLayer(Name="stdOut")];

% Assemble actor network
actorNetwork = dlnetwork;
actorNetwork = addLayers(actorNetwork,commonPath);
actorNetwork = addLayers(actorNetwork,meanPath);
actorNetwork = addLayers(actorNetwork,stdPath);

% Connect layers
actorNetwork = connectLayers(actorNetwork,"commOut","meanIn/in");
actorNetwork = connectLayers(actorNetwork,"commOut","stdIn/in");
actorNetwork = initialize(actorNetwork);

% Create Gaussian Actor
actor = rlContinuousGaussianActor(actorNetwork, observationInfo, actionInfo, ActionMeanOutputNames="meanOut",ActionStandardDeviationOutputNames="stdOut",ObservationInputNames="netObsIn");


%% Critic network
% Observation path
obsPath = [
    featureInputLayer(prod(observationInfo.Dimension), Name="obsPathIn")
    fullyConnectedLayer(128) 
    reluLayer
    fullyConnectedLayer(64, Name="obsPathOut")];

% Action path
actPath = [
    featureInputLayer(prod(actionInfo.Dimension), Name="actPathIn")
    fullyConnectedLayer(128)  
    reluLayer
    fullyConnectedLayer(64, Name="actPathOut")];

% Common path
commonPath = [
    concatenationLayer(1,2,Name="concat")
    reluLayer
    fullyConnectedLayer(1, Name="CriticOut")];  % Output layer


% Assemble critic network
criticNetwork = dlnetwork;
criticNetwork = addLayers(criticNetwork,obsPath);
criticNetwork = addLayers(criticNetwork,actPath);
criticNetwork = addLayers(criticNetwork,commonPath);

% Connect layers
criticNetwork = connectLayers(criticNetwork,"obsPathOut","concat/in1");
criticNetwork = connectLayers(criticNetwork,"actPathOut","concat/in2");

% Initialize two separate critic networks
criticNetwork1 = criticNetwork;
criticNetwork2 = criticNetwork;
criticNetwork1 = initialize(criticNetwork1);
criticNetwork2 = initialize(criticNetwork2);

% Create two critic Q-value functions
critic1 = rlQValueFunction(criticNetwork1, observationInfo, actionInfo, ActionInputNames="actPathIn", ObservationInputNames="obsPathIn");
critic2 = rlQValueFunction(criticNetwork2, observationInfo, actionInfo, ActionInputNames="actPathIn", ObservationInputNames="obsPathIn");


% Create SAC agent
actorOptimizerOptions = rlOptimizerOptions( ...
    Optimizer="adam", ...
    LearnRate=1e-5, ...
    GradientThreshold=1,...
    L2RegularizationFactor=1e-4);

criticOptimizerOptions = rlOptimizerOptions( ...
    Optimizer="adam", ...
    LearnRate=1e-4, ... 
    GradientThreshold=1,...
    L2RegularizationFactor=1e-3);

alphaOptimizerOptions = rl.option.EntropyWeightOptions( ...
      Optimizer="adam", ...
      LearnRate=1e-4, ...
      EntropyWeight=0.01,...
      TargetEntropy=-10,...
      GradientThreshold=1);


agentOptions = rlSACAgentOptions(...
    SampleTime=0.04,...
    DiscountFactor=0.99,...
    TargetSmoothFactor=1e-3,...
    ExperienceBufferLength=1e6,...
    MiniBatchSize=256,...
    NumWarmStartSteps=256*10,...
    ActorOptimizerOptions=actorOptimizerOptions, ...
    CriticOptimizerOptions=criticOptimizerOptions, ...
    EntropyWeightOptions=alphaOptimizerOptions);

agent = rlSACAgent(actor, [critic1, critic2], agentOptions);

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
