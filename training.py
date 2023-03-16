import gym
# import sys
import os
import datetime
# import random
# import cityflow
import numpy as np
import json
from gymCityflow import cityFlowGym
import shutil
# import GPyOpt
import optuna
from simpleCityflow import simpleCityflow

# from gym import spaces
# from stable_baselines3 import A2C
# from stable_baselines3 import DQN
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# # optuna attempt
# def objective(trial):
#     timestamp = datetime.datetime.now().strftime("%d-%b_%H%M")
#     logOutFolder = "output_logs" + "/optimizing_"+ timestamp
#     if not os.path.exists(logOutFolder):
#         os.makedirs(logOutFolder)
#     #open cityflow config file into dict
#     configDict = json.load(open("sample_data/jacob_config.json"))
#     configDict["dir"] = logOutFolder + '/'
#     with open("sample_data/jacob_config.json", "w") as outfile:
#         json.dump(configDict, outfile)

#     srcFlow = "sample_data/jacob_flow.json"
#     srcRoadnet = "sample_data/jacob_roadnet.json"
#     desFlow = logOutFolder + "/jacob_flow.json"
#     desRoadnet = logOutFolder + "/jacob_roadnet.json"
#     shutil.copy(srcFlow, desFlow)
#     shutil.copy(srcRoadnet, desRoadnet)
#     cityflowEnv = cityFlowGym("sample_data/jacob_config.json", 3600, logOutFolder, 10)
#     log_path = os.path.join('Optimizing', 'Logs')

#     n_steps = trial.suggest_categorical('n_steps', [32, 64, 128, 256])
#     gamma = trial.suggest_float('gamma', 0.9, 0.99)
#     learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
#     ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1)
#     vf_coef = trial.suggest_float('vf_coef', 0.0, 1.0)
#     max_grad_norm = trial.suggest_float('max_grad_norm', 0.1, 1.0)

#     model = PPO('MultiInputPolicy ', cityflowEnv, verbose=1, tensorboard_log=log_path,
#                 n_steps=n_steps, gamma=gamma,
#                 learning_rate=learning_rate, ent_coef=ent_coef,
#                 vf_coef=vf_coef, max_grad_norm=max_grad_norm)

#     model.learn(total_timesteps=36000)
#     models_dir = "models/PPO_36000_" + timestamp + "_optimizing"
#     model.save(f"{models_dir}")
#     # Evaluate the PPO agent
#     mean_reward = 0

#     timestamp = datetime.datetime.now().strftime("%d-%b_%H%M")
#     logOutFolder = "output_logs" + "/optimizingEval_"+ timestamp
#     if not os.path.exists(logOutFolder):
#         os.makedirs(logOutFolder)
#     #open cityflow config file into dict
#     configDict = json.load(open("sample_data/jacob_config.json"))
#     configDict["dir"] = logOutFolder + '/'
#     with open("sample_data/jacob_config.json", "w") as outfile:
#         json.dump(configDict, outfile)

#     srcFlow = "sample_data/jacob_flow.json"
#     srcRoadnet = "sample_data/jacob_roadnet.json"
#     desFlow = logOutFolder + "/jacob_flow.json"
#     desRoadnet = logOutFolder + "/jacob_roadnet.json"
#     shutil.copy(srcFlow, desFlow)
#     shutil.copy(srcRoadnet, desRoadnet)
#     cityflowEnv = cityFlowGym("sample_data/jacob_config.json", 3600, logOutFolder, 1)
    
#     for i in range(10):
#         obs = cityflowEnv.reset()
#         done = False
#         while not done:
#             action, _ = model.predict(obs)
#             obs, reward, done, _ = cityflowEnv.step(action)
#             mean_reward += reward
#     mean_reward /= 10
#     # Return the mean reward as the objective value
#     return mean_reward

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)

# # Print the best hyperparameters and objective value
# print('Best trial:')
# best_trial = study.best_trial
# print(f'  Value: {best_trial.value:.2f}')
# print('  Params:')
# for key, value in best_trial.params.items():
#     print(f'    {key}: {value}')

# with open("bestParams.json", "w") as outfile:
#     json.dump(best_trial.params, outfile)

# # try if can plot optuna graph
# if optuna.visualization.is_available():
#     try:
#         fig = optuna.visualization.plot_intermediate_values(study)
#         fig.write_image("images/plot_intermediate_values.png")
#     except Exception as e:
#         print("For plot_intermediate_values")
#         print(e)

#     try:
#         fig = optuna.visualization.plot_optimization_history(study)
#         fig.write_image("images/plot_optimization_history.png")
#     except Exception as e:
#         print("For plot_optimization_history")
#         print(e)

#     try:
#         fig = optuna.visualization.plot_param_importances(study)
#         fig.write_image("images/plot_param_importances.png")
#     except Exception as e:
#         print("For plot_param_importances")
#         print(e)


###############################################################################
# normal training

timestamp = datetime.datetime.now().strftime("%d-%b_%H%M")
logOutFolder = "output_logs" + "/training_"+ timestamp
if not os.path.exists(logOutFolder):
    os.makedirs(logOutFolder)
#open cityflow config file into dict
configDict = json.load(open("sample_data/jacob_config.json"))
configDict["dir"] = logOutFolder + '/'
with open("sample_data/jacob_config.json", "w") as outfile:
    json.dump(configDict, outfile)

srcFlow = "sample_data/jacob_flow.json"
srcRoadnet = "sample_data/jacob_roadnet.json"
desFlow = logOutFolder + "/jacob_flow.json"
desRoadnet = logOutFolder + "/jacob_roadnet.json"
shutil.copy(srcFlow, desFlow)
shutil.copy(srcRoadnet, desRoadnet)
cityflowEnv = simpleCityflow("sample_data/jacob_config.json", 3600, logOutFolder, 100)
log_path = os.path.join('Training', 'Logs')

# check_env(cityflowEnv, warn=True, skip_render_check=True)

model = PPO("MlpPolicy", cityflowEnv, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=360000)
models_dir = "models/PPO_360000_" + timestamp + "_simpleMix"
model.save(f"{models_dir}")