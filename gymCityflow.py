# import gym_cityflow
# from gym_cityflow.envs import Cityflow
import gym
# import sys
import os
# import random
import cityflow
import numpy as np
import json
import shutil
# import datetime
import math

from gym import spaces, logger
# from stable_baselines3 import A2C
# from stable_baselines3 import DQN
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.env_checker import check_env

class cityFlowGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config_path, episodeSteps, outputFolder):
        
        # edit config_path to change replayLogFile name for each step
#         with open(config_path, "r+") as jsonFile:
#             data = json.load(jsonFile)
#             fileName = data["replayLogFile"].split(".")[0]
#             data["replayLogFile"] = fileName + "_" + str(replayNumber) + ".txt"

#             jsonFile.seek(0)  # rewind
#             json.dump(data, jsonFile)
        
        # creating Cityflow engine
        self.engine = cityflow.Engine(config_path, thread_num=1)
        print("Initialising engine\n")
        
        #open cityflow config file into dict
        self.configDict = json.load(open(config_path))
        #open cityflow roadnet file into dict
        self.roadnetDict = json.load(open(self.configDict['dir'] + self.configDict['roadnetFile']))
        self.flowDict = json.load(open(self.configDict['dir'] + self.configDict['flowFile']))
        self.outputFolder = outputFolder
        
        #steps per episode
#         self.steps_per_episode = episodeSteps
        self.isDone = False
        self.currStep = 0
        self.maxStep = episodeSteps
        self.minLighphaseTime = 10 #approximate realistic time
        self.maxLighphaseTime = 60 #approximate realistic time
        self.maxSpeed = self.flowDict[0]["vehicle"]["maxSpeed"]
        self.info = {}
        self.maxLightPhase = 0
        self.allActionSpace = ""
        
        # create dict of controllable intersections and number of light phases
        self.intersections = {}
        for i in range(len(self.roadnetDict['intersections'])):
            # check if intersection is controllable
            if self.roadnetDict['intersections'][i]['virtual'] == False:
                currIntersection = self.roadnetDict['intersections'][i]
                
                # change maxLightPhase if needed for observationSpace definition
                # if self.maxLightPhase < len(currIntersection['trafficLight']['lightphases']):
                #     self.maxLightPhase = len(currIntersection['trafficLight']['lightphases'])
                
                
                # create incoming roadlink key that contains all incoming road lane id for each roadLink index
                roadLinkIntersectionDict = {}
                for roadLinkIndex, roadLink in enumerate(currIntersection["roadLinks"]):
                    startRoad = currIntersection["roadLinks"][roadLinkIndex]["startRoad"]
                    roadLinkSet = set()
                    for laneLinkIndex, laneLink in enumerate(currIntersection["roadLinks"][roadLinkIndex]["laneLinks"]):
                        # if laneLink["startLaneIndex"] != 5 and laneLink["startLaneIndex"] != 6:
                            tempLaneLink = startRoad + "_" + str(laneLink["startLaneIndex"])
                            roadLinkSet.add(tempLaneLink)
                    roadLinkIntersectionDict[roadLinkIndex] = roadLinkSet

                # debug roadLinkIntersectionDict
                # print("debug roadLinkIntersectionDict")
                # for key, value in roadLinkIntersectionDict.items():
                #     print(key, value)
                # print()
                    
                # using the roadLinkIntersectionDict, we create the incoming road lanes and store it to each lightphase    
                lightPhaseRoadLaneIntersectionDict = {}
                for lightPhase in range(len(currIntersection['trafficLight']['lightphases'])):
                    availableRoadLinks = set()
                    for roadLinkIndex in currIntersection['trafficLight']['lightphases'][lightPhase]["availableRoadLinks"]:
                        availableRoadLinks.update(roadLinkIntersectionDict[roadLinkIndex])
                    lightPhaseRoadLaneIntersectionDict[lightPhase] = availableRoadLinks

                # find sliproads (perm green light phase)
                if len(lightPhaseRoadLaneIntersectionDict) > 1:
                    slipRoads = lightPhaseRoadLaneIntersectionDict[0]
                    for link in range(1, len(lightPhaseRoadLaneIntersectionDict)):
                        slipRoads = slipRoads & lightPhaseRoadLaneIntersectionDict[link]
                # print("slipRoads: ", slipRoads)
                # print()
                # if there are sliproads, remove them
                if len(slipRoads) != 0:
                    for link in range(len(lightPhaseRoadLaneIntersectionDict)):
                        lightPhaseRoadLaneIntersectionDict[link] = lightPhaseRoadLaneIntersectionDict[link] - slipRoads

                # debug lightPhaseRoadLaneIntersectionDict
                # print("debug lightPhaseRoadLaneIntersectionDict")
                # for key, value in lightPhaseRoadLaneIntersectionDict.items():
                #     print(key, value)
                # print()

                # add intersection to dict where key = intersection_id
                # value = no of lightPhases, incoming lane names, outgoing lane names, directions for each lane group
                self.intersections[self.roadnetDict['intersections'][i]['id']] = { 
                    "lightPhases" : lightPhaseRoadLaneIntersectionDict,
                    "incomingRoadLinks" : roadLinkIntersectionDict
                }


        # print("self.intersections: ", self.intersections)
        
        #setup intersectionNames list for agent actions
        self.intersectionNames = []
        for key in self.intersections:
            self.intersectionNames.append(key)
        
        #define action space (num of lightPhases for each intersection) MultiDiscrete()
        # actionSpaceArray = []
        self.upperBound = []
        self.lowerBound = []
        smallestLightPhase = 0 # change smallestLightPhase of lightphase if needed
        for intersection in self.intersections:
            self.lowerBound.append(smallestLightPhase)
            self.lowerBound.append(self.minLighphaseTime)
            self.upperBound.append(len(self.intersections[intersection]["lightPhases"])-1)
            if len(self.intersections[intersection]["lightPhases"])-1 > self.maxLightPhase:
                self.maxLightPhase = len(self.intersections[intersection]["lightPhases"])-1
            self.upperBound.append(self.maxLighphaseTime)
#             lightphaseDurationSpace = self.minLighphaseTime = 0 #approximate realistic time
            
        self.action_space = spaces.Box(
            np.array([-1 for i in range(len(self.lowerBound))]).astype(np.float64), 
            np.array([1 for i in range(len(self.upperBound))]).astype(np.float64),
            dtype = np.float64
        )
        
        # define observation space
        numOfIntersections = len(self.intersectionNames)
        observationSpace = {
                "numVehicles" : spaces.Box(0, np.inf, (numOfIntersections, self.maxLightPhase+1), dtype=np.int32),
                "numWaitingVehicles" : spaces.Box(0, np.inf, (numOfIntersections, self.maxLightPhase+1), dtype=np.int32),
                "avgSpeed" : spaces.Box(0, np.inf, (numOfIntersections, self.maxLightPhase+1), dtype=np.float32)

            }
        
        self.observation_space = spaces.Dict(observationSpace)

    def step(self, action):
        actionArr = np.ndarray.tolist(action)
        # actionArr = [int(x) for x in actionArr]
        # converting action space to lightphase and lightphase duration
        for actionIndex in range(0, len(actionArr), 2):
            actionArr[actionIndex] =  min(math.floor((actionArr[actionIndex] + 1) / 2 * (self.upperBound[actionIndex]+1)), self.upperBound[actionIndex])
            actionArr[actionIndex+1] =  min(math.floor((actionArr[actionIndex+1] + 1) / 2 * (self.upperBound[actionIndex+1]-self.lowerBound[actionIndex+1]+1)) + self.lowerBound[actionIndex+1], self.upperBound[actionIndex+1])

        # print("actionArr:", actionArr)
        minTimer = 60
        negRewards = False # true means set negative infinity if lightphase action given is not part of the maxlightphases
        for intersection in range(len(self.intersectionNames)):
            # check if lightphase in intersection
            # print("Current intersection: ", self.intersectionNames[intersection])
            # print("actionArr[intersection*2]", actionArr[intersection*2])
            # print("len(self.intersections[self.intersectionNames[intersection]]['lightPhases'])", len(self.intersections[self.intersectionNames[intersection]]["lightPhases"]))
            if actionArr[intersection*2] < len(self.intersections[self.intersectionNames[intersection]]["lightPhases"]):
                self.engine.set_tl_phase(self.intersectionNames[intersection], int(actionArr[intersection*2]))
                if actionArr[1+intersection*2] < minTimer:
                    # print("Suggested timephase: ", actionArr[intersection*2])
                    # print("Suggested timephase duration: ", actionArr[1+ intersection*2])
                    # print()
                    minTimer = actionArr[1+ intersection*2]
                    
            else:
                negRewards = True
                
        # let scenario run for minTimer long
        if minTimer < 10:
            minTimer = 10
        
        for second in range(minTimer):
            self.engine.next_step()
            # print("going next step ", second)
        
        obs = self.get_observation()
        reward = self.get_reward(obs, negRewards)
        self.isDone = self.currStep >= self.maxStep
        info = obs
        self.currStep += minTimer
        self.allActionSpace += str(actionArr) + "\n"
        return obs, reward, self.isDone, info
    
    def get_observation(self):
        # get waiting vehicles for each lane first (key = laneID, value = numVehiclesWaiting)
        vehiclesWaitingByLaneDict = self.engine.get_lane_waiting_vehicle_count()
#         print(vehiclesWaitingByLaneDict)
        
        # get all vehicles speed (key = vehId, value = speed)
        vehiclesSpeedDict = self.engine.get_vehicle_speed()
#         print(vehiclesSpeedDict)
        
        # get all vehicles for each lane (key = laneId, value = [vehId])
        vehicleLaneDict = self.engine.get_lane_vehicles()
#         print(vehicleLaneDict)
        
        # create observation space for number of waiting vehicles and avgSpeed of moving+waiting vehicles of each lane
        # , for each lightphase
        
        # create for each roadlane first
        '''
        for roadLaneDict = {intersectionId : 
                                {roadLaneId: 
                                    {"numVehicles" : int,
                                     "numWaitingVehicles" : int,
                                     "avgSpeed" : float
                                                
                                    }
                
                                }
        
                            }
        '''
        roadLaneDict = {}
        for intersectionId, intersectionValue in self.intersections.items():
            roadLaneByIntersectionDict = {}
            
            # get a set of all incoming roadlanes for the intersection
            roadLaneByIntersectionSet = set()
            for roadLane in intersectionValue["incomingRoadLinks"].values():
                roadLaneByIntersectionSet.update(roadLane)
            
            # for each roadlane, find out num of waiting vehicles, and vehicles (waiting + nonwaiting) with speed
            for roadLane in roadLaneByIntersectionSet:
                tempRoadLaneDict = {}
                tempVehiclesArr = vehicleLaneDict[roadLane]
                tempRoadLaneDict["numVehicles"] = len(tempVehiclesArr)
                tempRoadLaneDict["numWaitingVehicles"] = vehiclesWaitingByLaneDict[roadLane]
                if len(tempVehiclesArr) == 0:
                    tempRoadLaneDict["avgSpeed"] = 0
                else:
                    tempAvgSpeed = 0
                    for vehicle in tempVehiclesArr:
                        tempAvgSpeed += vehiclesSpeedDict[vehicle]
                    tempRoadLaneDict["avgSpeed"] = tempAvgSpeed/len(tempVehiclesArr)
                
                roadLaneByIntersectionDict[roadLane] = tempRoadLaneDict
                
            # add intersection to roadLaneDict
            roadLaneDict[intersectionId] = roadLaneByIntersectionDict
        
        
        
        # define observation space
        observationSpace = {
                "numVehicles" : [],
                "numWaitingVehicles" : [],             
                "avgSpeed" : []
            }
        
        # for each intersection 
        for intersectionId, intersectionValue in self.intersections.items():
            numVehiclesArr = []
            numWaitingVehiclesArr = []             
            avgSpeedArr = []
            
            # for each lightphase in the intersection
            for lightPhase, roadLaneArr in intersectionValue["lightPhases"].items():
                totalVehiclesByPhase = 0
                waitingVehiclesByPhase = 0
                totalSpeedByPhase = 0
                
                # for each roadLane (availableRoadLinks) in each lightphase
                # print("roadLaneArr: ", roadLaneArr)
                # print("lightPhase: ", lightPhase)
                # break
                for roadLaneId in roadLaneArr:
                    # print("Calculated roadLaneId: ", roadLaneId)
                    totalVehiclesByPhase += roadLaneDict[intersectionId][roadLaneId]["numVehicles"]
                    waitingVehiclesByPhase += roadLaneDict[intersectionId][roadLaneId]["numWaitingVehicles"]
                    totalSpeedByPhase += (roadLaneDict[intersectionId][roadLaneId]["numVehicles"] * 
                                        roadLaneDict[intersectionId][roadLaneId]["avgSpeed"])
                    
                # error checking if theres no vehicles in the lane to eliminate divide by zero error
                if totalVehiclesByPhase == 0:
                    avgSpeedByPhase = 0
                else:
                    avgSpeedByPhase = totalSpeedByPhase/totalVehiclesByPhase
                    
                numVehiclesArr.append(totalVehiclesByPhase)
                numWaitingVehiclesArr.append(waitingVehiclesByPhase)
                avgSpeedArr.append(avgSpeedByPhase)
                
            # convert to np array
            tempNumVehicles = np.array(numVehiclesArr)
            tempNumWaitingVehicles = np.array(numWaitingVehiclesArr)
            tempAvgSpeed = np.array(avgSpeedArr)
            
            observationSpace["numVehicles"].append(tempNumVehicles)
            observationSpace["numWaitingVehicles"].append(tempNumWaitingVehicles)
            observationSpace["avgSpeed"].append(tempAvgSpeed)
                
        # convert to np array
        observationSpace["numVehicles"] = np.array(observationSpace["numVehicles"])
        observationSpace["numWaitingVehicles"] = np.array(observationSpace["numWaitingVehicles"])
        observationSpace["avgSpeed"] = np.array(observationSpace["avgSpeed"])

        return observationSpace
        
    def get_reward(self, observationSpace, negRewards):
        
        # if model returns lightphase that is not possible
        if negRewards == True:
            return -np.inf
        # aggregate total reward from all intersections
        totalReward = 0
        
        # numVehicles reward calc
        for intersection in observationSpace["numVehicles"]:
            for veh in intersection:
                
                totalReward -= veh
#                 print("veh", totalReward)
                
        # numWaitingVehicles reward calc
        for intersection in observationSpace["numWaitingVehicles"]:
            for waitingVeh in intersection:
                if waitingVeh == 0:
                        totalReward += 100
                else:
                    totalReward -= waitingVeh*100
#                 print("waitingVeh", totalReward)
                    
                        
        # avgSpeed reward calc
        for intersection in observationSpace["numWaitingVehicles"]:
            for speed in intersection:
                
                #reward if speed is
                if (speed-5) >= self.maxSpeed:
                    totalReward += 50
                else:
                    totalReward -= ((self.maxSpeed - speed)/self.maxSpeed)*50
#                 print("speed", totalReward)
        
            
        return totalReward/len(self.intersectionNames)
            
    def reset(self):
        # store action to text
        fileCount = self.move_file_with_counter()
        
        self.engine.reset()
        print("Engine has been reset\n")

        # dest_folder = self.outputFolder 

        # replayFilename = self.configDict["replayLogFile"]
        # split_replayFilename = os.path.splitext(replayFilename)
        # new_replayFilename = split_replayFilename[0] + '_' + str(fileCount) + split_replayFilename[1] 
        # # src_path = os.path.join(src_folder, replayFilename)
        # dest_path = os.path.join(dest_folder, new_replayFilename)
        # # # shutil.move(src_path, dest_path) # copy for replayLogFile
        # # with open(dest_path, 'w') as f:
        # #     pass
        # self.engine.set_replay_file(dest_path) # set replay file back to source file to create txt file
        # print("Replay file set to ", dest_path)
        # # self.engine.set_save_replay(True) # start saving replay

        # print("Count: ", fileCount)
        # print("Replay, roadnet log files, and allAction file have been moved to ", dest_folder)

        # change replaylog to another place
        self.create_new_replay(fileCount)
        self.currStep = 0
        obs = self.get_observation()
#         print("obs", obs)
        self.isDone = False
        return obs
    
    def create_new_replay(self, count):
        # self.engine.set_save_replay(False) # stop saving replay
        # src_folder = self.configDict["dir"].replace("/", "")
        dest_folder = self.outputFolder 

        replayFilename = self.configDict["replayLogFile"]
        split_replayFilename = os.path.splitext(replayFilename)
        replayFilename = split_replayFilename[0][:-1]
        new_replayFilename = replayFilename + str(count+1) + split_replayFilename[1] 
        # src_path = os.path.join(src_folder, replayFilename)
        dest_path = os.path.join(dest_folder, new_replayFilename)
        # # # shutil.move(src_path, dest_path) # copy for replayLogFile
        # with open(dest_path, 'w') as f:
        #     pass
        self.engine.set_replay_file(new_replayFilename) # set replay file back to source file to create txt file
        print("Replay file set to ", dest_path)
        # self.engine.set_save_replay(True) # start saving replay

        print("Count: ", count+1)
        print("Replay, roadnet log files, and allAction file have been moved to ", dest_folder)


    def move_file_with_counter(self):
        # self.engine.set_save_replay(False) # stop saving replay
        src_folder = self.configDict["dir"].replace("/", "")
        dest_folder = self.outputFolder 

        # create a single copy of roadnetLogFile in output_logs
        roadnetFilename = os.path.join(dest_folder, self.configDict["roadnetLogFile"])
        if (os.path.exists(roadnetFilename)):
            pass
        else:
            shutil.copyfile(os.path.join(src_folder, self.configDict["roadnetLogFile"]), roadnetFilename)
        
        allActions = "allActions.txt"
        counter = 0
        split_allActions = os.path.splitext(allActions)
        new_allActions = split_allActions[0] + '_' + str(counter) + split_allActions[1]
        while os.path.exists(os.path.join(dest_folder, new_allActions)):
            counter += 1
            new_allActions = split_allActions[0] + '_' + str(counter) + split_allActions[1]
        dest_path = os.path.join(dest_folder, new_allActions)
        with open(dest_path, 'w') as f:
            # write some text to the file
            f.write(self.allActionSpace)
        self.allActionSpace = ""
        return counter

        
        

    def render(self, mode='human', close=False):
        pass