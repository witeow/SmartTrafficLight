# import gym
# import sys
import os
# import random
# import cityflow
import numpy as np
import json
from gymCityflow import cityFlowGym
import datetime
import shutil

from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# for i in range(3600):
#     print("hello")
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
state = cityflowEnv.reset()

episodes = 3
for episode in range(1, episodes+1):
    
#     cityflowEnv.engine.reset()
    done = False
    score = 0
    step = 0

    # for phase in range(8):
    # action = np.array([4, 100])
    # n_state, reward, done, info = cityflowEnv.step(action)
    # print("action", action)
    # print("n_state", n_state)
    # print("reward", reward)
    # print("done", done)
    # print("info", info)
    # print("stepCount", cityflowEnv.currStep)
    # print()
    
    while not done:
#         env.render()
        action = cityflowEnv.action_space.sample()
        n_state, reward, done, info = cityflowEnv.step(action)
        print("action", action)
        print("n_state", n_state)
        print("reward", reward)
        print("done", done)
        print("info", info)
        print("stepCount", cityflowEnv.currStep)
        print()
        score += reward
        # break
#         break
    # score /= step
    state = cityflowEnv.reset()
        
    print('Episode:{} Score:{} with {} steps'.format(episode, score, step))