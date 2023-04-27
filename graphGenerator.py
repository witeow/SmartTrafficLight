# import gym
# import sys
import os
import datetime
# import random
# import cityflow
import numpy as np
# import json
# from gymCityflow import cityFlowGym
# from simpleCityflow import simpleCityflow
# import shutil
# import cityflow
import matplotlib.pyplot as plt
# import random

# from gym import spaces
# from stable_baselines3 import A2C
# from stable_baselines3 import DQN
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.env_checker import check_env

# comparing factors = numVehicles, numWaitingVehicles, totalWaitingTime, avgTravelTime
# variable = highTraffic, lowTraffic, mixTraffic (diff flow, roadnet, config)
# model 1 = lousy(high), model 2 = good(high), model 3 = good(low), model 4 = good(mixed)
def comparisonGraphGeneratorHighTraffic():
    recordInterval = [i for i in range(0, 3601, 100)]
    

    variable = "highTraffic"
    # load appropriate variable
    logOutFolder = "fyp_logs/" + variable
    if not os.path.exists(logOutFolder):
        os.makedirs(logOutFolder)
    srcFlow = "sample_data/"+variable+"_flow.json"
    srcRoadnet = "sample_data/"+variable+"_roadnet.json"
    desFlow = logOutFolder + "/"+variable+"_flow.json"
    desRoadnet = logOutFolder + "/"+variable+"_roadnet.json"
    shutil.copy(srcFlow, desFlow)
    shutil.copy(srcRoadnet, desRoadnet)
    
    #copy config file for cityFlowGymModel
    configDict = json.load(open("sample_data/"+variable+"_config.json"))
    configDict["dir"] = logOutFolder + '/'
    configDict["roadnetFile"] = variable + "_roadnet.json"
    configDict["flowFile"] = variable+"_flow.json"
    configDict["replayLogFile"] = "replayLog_cityFlowGymModel.txt"
    configDict["roadnetLogFile"] = "roadnetLog_cityFlowGymModel.txt"
    with open(logOutFolder+"/config_cityFlowGymModel.json", "w") as outfile:
        json.dump(configDict, outfile)

    cityflowEnv = cityFlowGym(logOutFolder+"/config_cityFlowGymModel.json", 3600, logOutFolder, 1)
    # load lousy model
    models_dir = "models/PPO_360000_11-Mar_1606_waiting+num+speed+prev"
    model = PPO.load(models_dir, cityflowEnv)
    dones = False

    cityflowGymNumVehicles = []
    cityflowGymNumWaitingVehicles = []
    cityflowGymWaitingTime = []
    cityflowGymAvgTravelTime = []
    obs = cityflowEnv.reset()
    prev = {}
    for i in range(3601):
        print(i)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = cityflowEnv.step(action)
        allSpeed = cityflowEnv.engine.get_vehicle_speed()
        curr = set()
        for veh, speed in allSpeed.items():
            if speed<0.1:
                curr.add(veh)
                if veh in prev:
                    prev[veh] += 1
                else:
                    prev[veh] = 1
        delete = set()
        for veh, count in prev.items():
            if veh not in curr:
                delete.add(veh)
        for veh in delete:
            del prev[veh]


        if i%100==0:
            cityflowGymNumVehicles.append(cityflowEnv.engine.get_vehicle_count())
            cityflowGymNumWaitingVehicles.append(len(prev))
            cityflowGymWaitingTime.append(sum(list(prev.values())))
            cityflowGymAvgTravelTime.append(cityflowEnv.engine.get_average_travel_time())

    #copy config file for simpleCityflow model
    configDict = json.load(open("sample_data/"+variable+"_config.json"))
    configDict["dir"] = logOutFolder + '/'
    configDict["roadnetFile"] = variable + "_roadnet.json"
    configDict["flowFile"] = variable+"_flow.json"
    configDict["replayLogFile"] = "replayLog_simpleCityflowModel.txt"
    configDict["roadnetLogFile"] = "roadnetLog_simpleCityflowModel.txt"
    with open(logOutFolder+"/config_simpleCityflowModel.json", "w") as outfile:
        json.dump(configDict, outfile)

    cityflowEnv = simpleCityflow(logOutFolder+"/config_simpleCityflowModel.json", 3600, logOutFolder, 1)
    # load good model
    models_dir = "models/PPO_360000_14-Mar_2204_simple"
    model = PPO.load(models_dir, cityflowEnv)
    dones = False

    simpleCityflowNumVehicles = []
    simpleCityflowNumWaitingVehicles = []
    simpleCityflowWaitingTime = []
    simpleCityflowAvgTravelTime = []
    obs = cityflowEnv.reset()
    prev = {}
    for i in range(3601):
        print(i)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = cityflowEnv.step(action)
        allSpeed = cityflowEnv.engine.get_vehicle_speed()
        curr = set()
        for veh, speed in allSpeed.items():
            if speed<0.1:
                curr.add(veh)
                if veh in prev:
                    prev[veh] += 1
                else:
                    prev[veh] = 1
        delete = set()
        for veh, count in prev.items():
            if veh not in curr:
                delete.add(veh)
        for veh in delete:
            del prev[veh]


        if i%100==0:
            simpleCityflowNumVehicles.append(cityflowEnv.engine.get_vehicle_count())
            simpleCityflowNumWaitingVehicles.append(len(prev))
            simpleCityflowWaitingTime.append(sum(list(prev.values())))
            simpleCityflowAvgTravelTime.append(cityflowEnv.engine.get_average_travel_time())

    #copy config file for nonRL model
    configDict = json.load(open("sample_data/"+variable+"_config.json"))
    configDict["dir"] = logOutFolder + '/'
    configDict["roadnetFile"] = variable + "_roadnet.json"
    configDict["flowFile"] = variable+"_flow.json"
    configDict["replayLogFile"] = "replayLog_nonRLModel.txt"
    configDict["roadnetLogFile"] = "roadnetLog_nonRLModel.txt"
    configDict["rlTrafficLight"] = False
    with open(logOutFolder+"/config_nonRLModel.json", "w") as outfile:
        json.dump(configDict, outfile)

    eng = cityflow.Engine(config_file=logOutFolder+"/config_nonRLModel.json", thread_num=1)
    # no model

    noRLModelNumVehicles = []
    noRLModelNumWaitingVehicles = []
    noRLModelWaitingTime = []
    noRLModelAvgTravelTime = []
    prev = {}

    for i in range(3601):
        print(i)
        eng.next_step()
        allSpeed = eng.get_vehicle_speed()
        curr = set()
        for veh, speed in allSpeed.items():
            if speed<0.1:
                curr.add(veh)
                if veh in prev:
                    prev[veh] += 1
                else:
                    prev[veh] = 1
        delete = set()
        for veh, count in prev.items():
            if veh not in curr:
                delete.add(veh)
        for veh in delete:
            del prev[veh]


        if i%100==0:
            noRLModelNumVehicles.append(eng.get_vehicle_count())
            noRLModelNumWaitingVehicles.append(len(prev))
            noRLModelWaitingTime.append(sum(list(prev.values())))
            noRLModelAvgTravelTime.append(eng.get_average_travel_time())


    # Plot the numVehicles
    plt.plot(recordInterval, cityflowGymNumVehicles, color='blue', label='cityFlowGym')
    plt.plot(recordInterval, simpleCityflowNumVehicles, color='red', label='simpleCityflow')
    plt.plot(recordInterval, noRLModelNumVehicles, color='green', label='No RL')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of Vehicles')

    # Display the plot
    plt.savefig(logOutFolder+"/Number of Vehicles.png")
    plt.clf()

    # Plot the waitingNumVehicles
    plt.plot(recordInterval, cityflowGymNumWaitingVehicles, color='blue', label='cityFlowGym')
    plt.plot(recordInterval, simpleCityflowNumWaitingVehicles, color='red', label='simpleCityflow')
    plt.plot(recordInterval, noRLModelNumWaitingVehicles, color='green', label='No RL')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of Waiting Vehicles')

    # Display the plot
    plt.savefig(logOutFolder+"/Number of Waiting Vehicles.png")
    plt.clf()

    # Plot the waiting time
    plt.plot(recordInterval, cityflowGymWaitingTime, color='blue', label='cityFlowGym')
    plt.plot(recordInterval, simpleCityflowWaitingTime, color='red', label='simpleCityflow')
    plt.plot(recordInterval, noRLModelWaitingTime, color='green', label='No RL')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Total waiting time')

    # Display the plot
    plt.savefig(logOutFolder+"/Total Waiting Time.png")
    plt.clf()

    # Plot the avg travel time
    plt.plot(recordInterval, cityflowGymAvgTravelTime, color='blue', label='cityFlowGym')
    plt.plot(recordInterval, simpleCityflowAvgTravelTime, color='red', label='simpleCityflow')
    plt.plot(recordInterval, noRLModelAvgTravelTime, color='green', label='No RL')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Average Travel Time')

    # Display the plot
    plt.savefig(logOutFolder+"/Average Travel Time.png")
    plt.clf()

def comparisonGraphGeneratorLowTraffic():
    recordInterval = [i for i in range(0, 3601, 100)]
    

    variable = "lowTraffic"
    # load appropriate variable
    logOutFolder = "fyp_logs/" + variable
    if not os.path.exists(logOutFolder):
        os.makedirs(logOutFolder)
    srcFlow = "sample_data/"+variable+"_flow.json"
    srcRoadnet = "sample_data/"+variable+"_roadnet.json"
    desFlow = logOutFolder + "/"+variable+"_flow.json"
    desRoadnet = logOutFolder + "/"+variable+"_roadnet.json"
    shutil.copy(srcFlow, desFlow)
    shutil.copy(srcRoadnet, desRoadnet)
    
    #copy config file for cityFlowGymModel
    configDict = json.load(open("sample_data/"+variable+"_config.json"))
    configDict["dir"] = logOutFolder + '/'
    configDict["roadnetFile"] = variable + "_roadnet.json"
    configDict["flowFile"] = variable+"_flow.json"
    configDict["replayLogFile"] = "replayLog_cityFlowGymModel.txt"
    configDict["roadnetLogFile"] = "roadnetLog_cityFlowGymModel.txt"
    with open(logOutFolder+"/config_cityFlowGymModel.json", "w") as outfile:
        json.dump(configDict, outfile)

    cityflowEnv = cityFlowGym(logOutFolder+"/config_cityFlowGymModel.json", 3600, logOutFolder, 1)
    # load lousy model
    models_dir = "models/PPO_360000_11-Mar_1606_waiting+num+speed+prev"
    model = PPO.load(models_dir, cityflowEnv)
    dones = False

    cityflowGymNumVehicles = []
    cityflowGymNumWaitingVehicles = []
    cityflowGymWaitingTime = []
    cityflowGymAvgTravelTime = []
    obs = cityflowEnv.reset()
    prev = {}
    for i in range(3601):
        print(i)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = cityflowEnv.step(action)
        allSpeed = cityflowEnv.engine.get_vehicle_speed()
        curr = set()
        for veh, speed in allSpeed.items():
            if speed<0.1:
                curr.add(veh)
                if veh in prev:
                    prev[veh] += 1
                else:
                    prev[veh] = 1
        delete = set()
        for veh, count in prev.items():
            if veh not in curr:
                delete.add(veh)
        for veh in delete:
            del prev[veh]


        if i%100==0:
            cityflowGymNumVehicles.append(cityflowEnv.engine.get_vehicle_count())
            cityflowGymNumWaitingVehicles.append(len(prev))
            cityflowGymWaitingTime.append(sum(list(prev.values())))
            cityflowGymAvgTravelTime.append(cityflowEnv.engine.get_average_travel_time())

    #copy config file for simpleCityflow model
    configDict = json.load(open("sample_data/"+variable+"_config.json"))
    configDict["dir"] = logOutFolder + '/'
    configDict["roadnetFile"] = variable + "_roadnet.json"
    configDict["flowFile"] = variable+"_flow.json"
    configDict["replayLogFile"] = "replayLog_simpleCityflowModel.txt"
    configDict["roadnetLogFile"] = "roadnetLog_simpleCityflowModel.txt"
    with open(logOutFolder+"/config_simpleCityflowModel.json", "w") as outfile:
        json.dump(configDict, outfile)

    cityflowEnv = simpleCityflow(logOutFolder+"/config_simpleCityflowModel.json", 3600, logOutFolder, 1)
    # load good model
    models_dir = "models/PPO_360000_15-Mar_1145_simpleSlow"
    model = PPO.load(models_dir, cityflowEnv)
    dones = False

    simpleCityflowNumVehicles = []
    simpleCityflowNumWaitingVehicles = []
    simpleCityflowWaitingTime = []
    simpleCityflowAvgTravelTime = []
    obs = cityflowEnv.reset()
    prev = {}
    for i in range(3601):
        print(i)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = cityflowEnv.step(action)
        allSpeed = cityflowEnv.engine.get_vehicle_speed()
        curr = set()
        for veh, speed in allSpeed.items():
            if speed<0.1:
                curr.add(veh)
                if veh in prev:
                    prev[veh] += 1
                else:
                    prev[veh] = 1
        delete = set()
        for veh, count in prev.items():
            if veh not in curr:
                delete.add(veh)
        for veh in delete:
            del prev[veh]


        if i%100==0:
            simpleCityflowNumVehicles.append(cityflowEnv.engine.get_vehicle_count())
            simpleCityflowNumWaitingVehicles.append(len(prev))
            simpleCityflowWaitingTime.append(sum(list(prev.values())))
            simpleCityflowAvgTravelTime.append(cityflowEnv.engine.get_average_travel_time())

    #copy config file for nonRL model
    configDict = json.load(open("sample_data/"+variable+"_config.json"))
    configDict["dir"] = logOutFolder + '/'
    configDict["roadnetFile"] = variable + "_roadnet.json"
    configDict["flowFile"] = variable+"_flow.json"
    configDict["replayLogFile"] = "replayLog_nonRLModel.txt"
    configDict["roadnetLogFile"] = "roadnetLog_nonRLModel.txt"
    configDict["rlTrafficLight"] = False
    with open(logOutFolder+"/config_nonRLModel.json", "w") as outfile:
        json.dump(configDict, outfile)

    eng = cityflow.Engine(config_file=logOutFolder+"/config_nonRLModel.json", thread_num=1)
    # no model

    noRLModelNumVehicles = []
    noRLModelNumWaitingVehicles = []
    noRLModelWaitingTime = []
    noRLModelAvgTravelTime = []
    prev = {}

    for i in range(3601):
        print(i)
        eng.next_step()
        allSpeed = eng.get_vehicle_speed()
        curr = set()
        for veh, speed in allSpeed.items():
            if speed<0.1:
                curr.add(veh)
                if veh in prev:
                    prev[veh] += 1
                else:
                    prev[veh] = 1
        delete = set()
        for veh, count in prev.items():
            if veh not in curr:
                delete.add(veh)
        for veh in delete:
            del prev[veh]


        if i%100==0:
            noRLModelNumVehicles.append(eng.get_vehicle_count())
            noRLModelNumWaitingVehicles.append(len(prev))
            noRLModelWaitingTime.append(sum(list(prev.values())))
            noRLModelAvgTravelTime.append(eng.get_average_travel_time())


    # Plot the numVehicles
    plt.plot(recordInterval, cityflowGymNumVehicles, color='blue', label='cityFlowGym')
    plt.plot(recordInterval, simpleCityflowNumVehicles, color='red', label='simpleCityflow')
    plt.plot(recordInterval, noRLModelNumVehicles, color='green', label='No RL')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of Vehicles')

    # Display the plot
    plt.savefig(logOutFolder+"/Number of Vehicles.png")
    plt.clf()

    # Plot the waitingNumVehicles
    plt.plot(recordInterval, cityflowGymNumWaitingVehicles, color='blue', label='cityFlowGym')
    plt.plot(recordInterval, simpleCityflowNumWaitingVehicles, color='red', label='simpleCityflow')
    plt.plot(recordInterval, noRLModelNumWaitingVehicles, color='green', label='No RL')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of Waiting Vehicles')

    # Display the plot
    plt.savefig(logOutFolder+"/Number of Waiting Vehicles.png")
    plt.clf()

    # Plot the waiting time
    plt.plot(recordInterval, cityflowGymWaitingTime, color='blue', label='cityFlowGym')
    plt.plot(recordInterval, simpleCityflowWaitingTime, color='red', label='simpleCityflow')
    plt.plot(recordInterval, noRLModelWaitingTime, color='green', label='No RL')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Total waiting time')

    # Display the plot
    plt.savefig(logOutFolder+"/Total Waiting Time.png")
    plt.clf()

    # Plot the avg travel time
    plt.plot(recordInterval, cityflowGymAvgTravelTime, color='blue', label='cityFlowGym')
    plt.plot(recordInterval, simpleCityflowAvgTravelTime, color='red', label='simpleCityflow')
    plt.plot(recordInterval, noRLModelAvgTravelTime, color='green', label='No RL')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Average Travel Time')

    # Display the plot
    plt.savefig(logOutFolder+"/Average Travel Time.png")
    plt.clf()

def comparisonGraphGeneratorMixTraffic():
    recordInterval = [i for i in range(0, 3601, 100)]
    

    variable = "mixTraffic"
    # load appropriate variable
    logOutFolder = "fyp_logs/" + variable
    if not os.path.exists(logOutFolder):
        os.makedirs(logOutFolder)
    srcFlow = "sample_data/"+variable+"_flow.json"
    srcRoadnet = "sample_data/"+variable+"_roadnet.json"
    desFlow = logOutFolder + "/"+variable+"_flow.json"
    desRoadnet = logOutFolder + "/"+variable+"_roadnet.json"
    shutil.copy(srcFlow, desFlow)
    shutil.copy(srcRoadnet, desRoadnet)
    
    #copy config file for cityFlowGymModel
    configDict = json.load(open("sample_data/"+variable+"_config.json"))
    configDict["dir"] = logOutFolder + '/'
    configDict["roadnetFile"] = variable + "_roadnet.json"
    configDict["flowFile"] = variable+"_flow.json"
    configDict["replayLogFile"] = "replayLog_cityFlowGymModel.txt"
    configDict["roadnetLogFile"] = "roadnetLog_cityFlowGymModel.txt"
    with open(logOutFolder+"/config_cityFlowGymModel.json", "w") as outfile:
        json.dump(configDict, outfile)

    cityflowEnv = cityFlowGym(logOutFolder+"/config_cityFlowGymModel.json", 3600, logOutFolder, 1)
    # load lousy model
    models_dir = "models/PPO_360000_11-Mar_1606_waiting+num+speed+prev"
    model = PPO.load(models_dir, cityflowEnv)
    dones = False

    cityflowGymNumVehicles = []
    cityflowGymNumWaitingVehicles = []
    cityflowGymWaitingTime = []
    cityflowGymAvgTravelTime = []
    obs = cityflowEnv.reset()
    prev = {}
    for i in range(3601):
        print(i)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = cityflowEnv.step(action)
        allSpeed = cityflowEnv.engine.get_vehicle_speed()
        curr = set()
        for veh, speed in allSpeed.items():
            if speed<0.1:
                curr.add(veh)
                if veh in prev:
                    prev[veh] += 1
                else:
                    prev[veh] = 1
        delete = set()
        for veh, count in prev.items():
            if veh not in curr:
                delete.add(veh)
        for veh in delete:
            del prev[veh]


        if i%100==0:
            cityflowGymNumVehicles.append(cityflowEnv.engine.get_vehicle_count())
            cityflowGymNumWaitingVehicles.append(len(prev))
            cityflowGymWaitingTime.append(sum(list(prev.values())))
            cityflowGymAvgTravelTime.append(cityflowEnv.engine.get_average_travel_time())

    #copy config file for simpleCityflow model
    configDict = json.load(open("sample_data/"+variable+"_config.json"))
    configDict["dir"] = logOutFolder + '/'
    configDict["roadnetFile"] = variable + "_roadnet.json"
    configDict["flowFile"] = variable+"_flow.json"
    configDict["replayLogFile"] = "replayLog_simpleCityflowModel.txt"
    configDict["roadnetLogFile"] = "roadnetLog_simpleCityflowModel.txt"
    with open(logOutFolder+"/config_simpleCityflowModel.json", "w") as outfile:
        json.dump(configDict, outfile)

    cityflowEnv = simpleCityflow(logOutFolder+"/config_simpleCityflowModel.json", 3600, logOutFolder, 1)
    # load good model
    models_dir = "models/PPO_360000_15-Mar_1446_simpleMix"
    model = PPO.load(models_dir, cityflowEnv)
    dones = False

    simpleCityflowNumVehicles = []
    simpleCityflowNumWaitingVehicles = []
    simpleCityflowWaitingTime = []
    simpleCityflowAvgTravelTime = []
    obs = cityflowEnv.reset()
    prev = {}
    for i in range(3601):
        print(i)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = cityflowEnv.step(action)
        allSpeed = cityflowEnv.engine.get_vehicle_speed()
        curr = set()
        for veh, speed in allSpeed.items():
            if speed<0.1:
                curr.add(veh)
                if veh in prev:
                    prev[veh] += 1
                else:
                    prev[veh] = 1
        delete = set()
        for veh, count in prev.items():
            if veh not in curr:
                delete.add(veh)
        for veh in delete:
            del prev[veh]


        if i%100==0:
            simpleCityflowNumVehicles.append(cityflowEnv.engine.get_vehicle_count())
            simpleCityflowNumWaitingVehicles.append(len(prev))
            simpleCityflowWaitingTime.append(sum(list(prev.values())))
            simpleCityflowAvgTravelTime.append(cityflowEnv.engine.get_average_travel_time())

    #copy config file for nonRL model
    configDict = json.load(open("sample_data/"+variable+"_config.json"))
    configDict["dir"] = logOutFolder + '/'
    configDict["roadnetFile"] = variable + "_roadnet.json"
    configDict["flowFile"] = variable+"_flow.json"
    configDict["replayLogFile"] = "replayLog_nonRLModel.txt"
    configDict["roadnetLogFile"] = "roadnetLog_nonRLModel.txt"
    configDict["rlTrafficLight"] = False
    with open(logOutFolder+"/config_nonRLModel.json", "w") as outfile:
        json.dump(configDict, outfile)

    eng = cityflow.Engine(config_file=logOutFolder+"/config_nonRLModel.json", thread_num=1)
    # no model

    noRLModelNumVehicles = []
    noRLModelNumWaitingVehicles = []
    noRLModelWaitingTime = []
    noRLModelAvgTravelTime = []
    prev = {}

    for i in range(3601):
        print(i)
        eng.next_step()
        allSpeed = eng.get_vehicle_speed()
        curr = set()
        for veh, speed in allSpeed.items():
            if speed<0.1:
                curr.add(veh)
                if veh in prev:
                    prev[veh] += 1
                else:
                    prev[veh] = 1
        delete = set()
        for veh, count in prev.items():
            if veh not in curr:
                delete.add(veh)
        for veh in delete:
            del prev[veh]


        if i%100==0:
            noRLModelNumVehicles.append(eng.get_vehicle_count())
            noRLModelNumWaitingVehicles.append(len(prev))
            noRLModelWaitingTime.append(sum(list(prev.values())))
            noRLModelAvgTravelTime.append(eng.get_average_travel_time())


    # Plot the numVehicles
    plt.plot(recordInterval, cityflowGymNumVehicles, color='blue', label='cityFlowGym')
    plt.plot(recordInterval, simpleCityflowNumVehicles, color='red', label='simpleCityflow')
    plt.plot(recordInterval, noRLModelNumVehicles, color='green', label='No RL')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of Vehicles')

    # Display the plot
    plt.savefig(logOutFolder+"/Number of Vehicles.png")
    plt.clf()

    # Plot the waitingNumVehicles
    plt.plot(recordInterval, cityflowGymNumWaitingVehicles, color='blue', label='cityFlowGym')
    plt.plot(recordInterval, simpleCityflowNumWaitingVehicles, color='red', label='simpleCityflow')
    plt.plot(recordInterval, noRLModelNumWaitingVehicles, color='green', label='No RL')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of Waiting Vehicles')

    # Display the plot
    plt.savefig(logOutFolder+"/Number of Waiting Vehicles.png")
    plt.clf()

    # Plot the waiting time
    plt.plot(recordInterval, cityflowGymWaitingTime, color='blue', label='cityFlowGym')
    plt.plot(recordInterval, simpleCityflowWaitingTime, color='red', label='simpleCityflow')
    plt.plot(recordInterval, noRLModelWaitingTime, color='green', label='No RL')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Total waiting time')

    # Display the plot
    plt.savefig(logOutFolder+"/Total Waiting Time.png")
    plt.clf()

    # Plot the avg travel time
    plt.plot(recordInterval, cityflowGymAvgTravelTime, color='blue', label='cityFlowGym')
    plt.plot(recordInterval, simpleCityflowAvgTravelTime, color='red', label='simpleCityflow')
    plt.plot(recordInterval, noRLModelAvgTravelTime, color='green', label='No RL')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Average Travel Time')

    # Display the plot
    plt.savefig(logOutFolder+"/Average Travel Time.png")
    plt.clf()

# comparisonGraphGeneratorHighTraffic()
# comparisonGraphGeneratorLowTraffic()
# comparisonGraphGeneratorMixTraffic()

def trainingTimeGraphGenerator():
    recordInterval = [i for i in range(0,30,3)]
    cityflowTrainingTime = [1.254, 1.159, 0.582, 0.599, 0.498, 0.723, 0.903, 0.873, 0.796, 1.019]
    trafficTrainingTime = [4.281, 4.214, 3.877, 3.915, 4.102, 4.274, 3.836, 3.581, 3.901, 4.120]

    # Plot the avg travel time
    plt.plot(recordInterval, cityflowTrainingTime, color='brown', label='Cityflow')
    plt.plot(recordInterval, trafficTrainingTime, color='gray', label='Traffic3d')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Training Time')

    # Display the plot
    plt.savefig("Time taken per epoch.png")
    plt.clf()

trainingTimeGraphGenerator()