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

# from gym import spaces
# from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.env_checker import check_env


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
cityflowEnv = cityFlowGym("sample_data/jacob_config.json", 3600, logOutFolder)
log_path = os.path.join('Training', 'Logs')



model = PPO("MultiInputPolicy", cityflowEnv, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=360000)
models_dir = "models/PPO_360000"
model.save(f"{models_dir}")