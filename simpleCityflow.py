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

class simpleCityflow(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config_path, episodeSteps, outputFolder, recordInterval):
        
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
        self.resetCount = 0
        self.resetInterval = recordInterval
        
        #steps per episode
#         self.steps_per_episode = episodeSteps
        self.isDone = False
        self.currStep = 0
        self.maxStep = episodeSteps
        self.minLighphaseTime = 10 #approximate realistic time
        self.maxLighphaseTime = 30 #approximate realistic time
        self.maxSpeed = self.flowDict[0]["vehicle"]["maxSpeed"]
        self.info = {}
        self.maxLightPhase = 0
        self.allActionSpace = ""
        self.prevWaitingVehicles = {} # vehId : how many actions round waited
        self.newPrevWaitingVehicles = {}


        
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
            # self.lowerBound.append(smallestLightPhase)
            # self.lowerBound.append(self.minLighphaseTime)
            self.upperBound.append(len(self.intersections[intersection]["lightPhases"])-1)
            if len(self.intersections[intersection]["lightPhases"])-1 > self.maxLightPhase:
                self.maxLightPhase = len(self.intersections[intersection]["lightPhases"])-1
            self.upperBound.append(5) # 0 - 5 = 10 - 60  
#             lightphaseDurationSpace = self.minLighphaseTime = 0 #approximate realistic time
            
        self.action_space = spaces.MultiDiscrete(np.array(self.upperBound))
        
        # define observation space
        numOfIntersections = len(self.intersectionNames)
        observationSpace = spaces.Box(0, np.inf, (numOfIntersections, self.maxLightPhase+1), dtype=np.int32)
        self.observation_space = observationSpace   
            # {
            #     "numVehicles" : spaces.Box(0, np.inf, (numOfIntersections, self.maxLightPhase+1), dtype=np.int32),
            #     "numWaitingVehicles" : spaces.Box(0, np.inf, (numOfIntersections, self.maxLightPhase+1), dtype=np.int32),
            #     "avgSpeed" : spaces.Box(0, np.inf, (numOfIntersections, self.maxLightPhase+1), dtype=np.float32),
            #     "prevWaitingVehicles" : spaces.Box(0, np.inf, (numOfIntersections, self.maxLightPhase+1), dtype=np.float32)
            #     spaces.Box(0, np.inf, (numOfIntersections, self.maxLightPhase+1), dtype=np.float32)
            # }
        
        # self.observation_space = spaces.Dict(observationSpace)

    def step(self, action):
        print("action:", action)
        actionArr = np.ndarray.tolist(action)
        # print("Old actionArr:", actionArr)
        # actionArr = [int(x) for x in actionArr]
        # converting action space to lightphase and lightphase duration
        # for actionIndex in range(0, len(actionArr), 2):
        #     actionArr[actionIndex] =  min(math.floor((actionArr[actionIndex] + 1) / 2 * (self.upperBound[actionIndex]+1)), self.upperBound[actionIndex])
        #     actionArr[actionIndex+1] =  min(math.floor((actionArr[actionIndex+1] + 1) / 2 * (self.upperBound[actionIndex+1]-self.lowerBound[actionIndex+1]+1)) + self.lowerBound[actionIndex+1], self.upperBound[actionIndex+1])

        print("actionArr:", actionArr)
        minTimer = actionArr[1]*10 + 10
        # negRewards = False # true means set negative infinity if lightphase action given is not part of the maxlightphases
        for intersection in range(len(self.intersectionNames)):
            # check if lightphase in intersection
            # print("Current intersection: ", self.intersectionNames[intersection])
            # print("actionArr[intersection*2]", actionArr[intersection*2])
            # print("len(self.intersections[self.intersectionNames[intersection]]['lightPhases'])", len(self.intersections[self.intersectionNames[intersection]]["lightPhases"]))
            # if actionArr[intersection*2] < len(self.intersections[self.intersectionNames[intersection]]["lightPhases"]):
            # self.engine.set_tl_phase(self.intersectionNames[intersection], int(actionArr[intersection*2]))
            self.engine.set_tl_phase(self.intersectionNames[intersection], actionArr[intersection*2])
            minTimer = min(minTimer, actionArr[intersection*2+1]*10 + 10)
            print("Suggested timephase: ", actionArr[intersection*2])
            print("Suggested timephase duration: ", actionArr[intersection*2+1]*10 + 10)
                # if actionArr[1+intersection*2] < minTimer:
                #     # print("Suggested timephase: ", actionArr[intersection*2])
                #     # print("Suggested timephase duration: ", actionArr[1+ intersection*2])
                #     # print()
                #     minTimer = actionArr[1+ intersection*2]
                    
            # else:
            #     negRewards = True
                
        # let scenario run for minTimer long
        # if minTimer < self.minLighphaseTime:
        #     minTimer = 10
        
        for second in range(minTimer):
            self.engine.next_step()
            # print("going next step ", second)
        
        self.prevWaitingVehicles = self.newPrevWaitingVehicles

        obs = self.get_observation()
        reward = self.get_reward(obs, actionArr)
        self.isDone = self.currStep >= self.maxStep
        info = {}
        self.currStep += minTimer
        self.allActionSpace += str(actionArr) + "\n"
        print("obs", obs)
        print("reward", reward)
        print()
        # print("isDone", self.isDone)
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
        
        # to find throughput, we use self.prevWaitingVehicles and check which vehs are still inside, vehId : actions waited
        # if veh waited for 1 round = 1
        # if veh waited for 2 round = 2
        # if veh waited for 3 rounds = 4 (2**2)


        # create observation space for number of waiting vehicles and avgSpeed of moving+waiting vehicles of each lane
        # , for each lightphase
        
        # create for each roadlane first
        '''
        for roadLaneDict = {intersectionId : 
                                {roadLaneId: 
                                     "prevWaitingVehicles" :int
                                                
                                    }
                
                                }
        
                            }
        '''
        roadLaneDict = {}
        self.newPrevWaitingVehicles = {}
        for intersectionId, intersectionValue in self.intersections.items():
            roadLaneByIntersectionDict = {}
            newPrevWaitingVehiclesIntersection = {}
            # get a set of all incoming roadlanes for the intersection
            roadLaneByIntersectionSet = set()
            for roadLane in intersectionValue["incomingRoadLinks"].values():
                roadLaneByIntersectionSet.update(roadLane)
            
            # for each roadlane, find out num of waiting vehicles, and vehicles (waiting + nonwaiting) and speed
            for roadLane in roadLaneByIntersectionSet:
                tempRoadLaneDict = {}

                # calc prevWaitingVehicles
                tempRoadLaneDict["prevWaitingVehicles"] = vehiclesWaitingByLaneDict[roadLane]
                newPrevWaitingVehiclesLane = {}

                # see which veh id is waiting
                for veh in vehicleLaneDict[roadLane]:
                    if vehiclesSpeedDict[veh] < 0.1:
                        # tempRoadLaneDict["prevWaitingVehicles"] += 1
                        if veh not in self.prevWaitingVehicles:
                            newPrevWaitingVehiclesLane[veh] = 1
                        else:
                            newPrevWaitingVehiclesLane[veh] = self.prevWaitingVehicles[veh] + 1
                            

                # update on a lane level
                newPrevWaitingVehiclesIntersection.update(newPrevWaitingVehiclesLane)
                
                roadLaneByIntersectionDict[roadLane] = tempRoadLaneDict
                
            # add intersection to roadLaneDict
            roadLaneDict[intersectionId] = roadLaneByIntersectionDict

            # update on an intersection level
            self.newPrevWaitingVehicles.update(newPrevWaitingVehiclesIntersection)

        # # update on a whole env level
        # self.prevWaitingVehicles = newPrevWaitingVehicles
        
        # define observation space
        observationSpace = []
        
        # for each intersection 
        for intersectionId, intersectionValue in self.intersections.items():
            prevWaitingVehiclesArr = []
            
            # for each lightphase in the intersection
            for lightPhase, roadLaneArr in intersectionValue["lightPhases"].items():
                prevWaitingVehiclesByPhase = 0
                
                # for each roadLane (availableRoadLinks) in each lightphase
                # print("roadLaneArr: ", roadLaneArr)
                # print("lightPhase: ", lightPhase)
                # break
                for roadLaneId in roadLaneArr:
                    # print("Calculated roadLaneId: ", roadLaneId)
                    prevWaitingVehiclesByPhase += roadLaneDict[intersectionId][roadLaneId]["prevWaitingVehicles"]
                
                prevWaitingVehiclesArr.append(prevWaitingVehiclesByPhase)
                
            # convert to np array
            observationSpace.append(np.array(prevWaitingVehiclesArr))
                
        # convert to np array
        observationSpace = np.array(observationSpace)

        # print("observationSpace", observationSpace)
        return observationSpace
        
    def get_reward(self, observationSpace, actionSpace):
        
        # aggregate total reward from all intersections
        totalReward = 0
        
        observationSpace = np.ndarray.tolist(observationSpace)
        
        print("self.preWaitingSort", self.preWaitingSort)
        # prevWaitingVehicles reward calc
        for intersectionIndex, intersection in enumerate(observationSpace):
            # intersectionAction = actionSpace[intersectionIndex]

            # print("self.preWaitingSort", self.preWaitingSort)
            if len(self.preWaitingSort) == 0:
                actionIndex = 0
            else:
                for index, lightPhase in enumerate(self.preWaitingSort[intersectionIndex]):
                    # print("lightPhase", lightPhase)
                    if lightPhase[0] == actionSpace[intersectionIndex*2]:
                        actionIndex = index

            # print("self.preWaitingSort", self.preWaitingSort)

            vehLeft = 0

            if len(self.newPrevWaitingVehicles) != 0:
                vehRemaining = sum(list(self.newPrevWaitingVehicles.values()))
            else:
                vehRemaining = 1e200

            oldTotalVeh = max(len(self.prevWaitingVehicles),1)
            for oldVeh in self.prevWaitingVehicles:
                if oldVeh not in self.newPrevWaitingVehicles:
                    vehLeft += 1
            print("reward 1st part: ", (1/(1+actionIndex))* 100)
            print("reward actionIndex: ", actionIndex)
            print("reward vehLeft: ", vehLeft)
            print("reward oldTotalVeh: ", oldTotalVeh)
            print("reward vehRemaining: ", vehRemaining)
            print("reward 4th (2 and 3) part: ", ((vehLeft/oldTotalVeh) + (1/vehRemaining))*50)
            totalReward += (1/(1+actionIndex))* 100 + ((vehLeft/oldTotalVeh) +(1/vehRemaining))*50

        self.preWaitingSort = []
        for intersectionIndex, intersection in enumerate(observationSpace):
            preWaitingIntersection = []
            for index, prevWaitingVehicle in enumerate(intersection):
                preWaitingIntersection.append(tuple([index, prevWaitingVehicle]))
            preWaitingIntersection.sort(key=lambda x: x[1], reverse=True)
            self.preWaitingSort.append(preWaitingIntersection)

        print("totalReward/len(self.intersectionNames): ", totalReward/len(self.intersectionNames))
        # print("self.intersectionNames", self.intersectionNames)
        return totalReward/len(self.intersectionNames)
            
    def reset(self):
        # store action to text
        if(self.resetCount%self.resetInterval==0):
            self.move_file_with_counter()
        self.allActionSpace = ""
        self.resetCount += 1
        # fileCount = self.move_file_with_counter()
        # if(self.resetCount%2==0):
        #     self.move_file_with_counter()
        # self.allActionSpace = ""
        
        self.engine.reset()
        print("Engine has been reset\n")

        # dest_folder = self.outputFolder 

        # change replaylog to another place
        # self.create_new_replay(fileCount)
        if((self.resetCount)%self.resetInterval==0):
            self.engine.set_save_replay(True)
            self.create_new_replay()
        else:
            self.engine.set_save_replay(False)
        self.currStep = 0
        obs = self.get_observation()
        # print("obs", obs)
        self.isDone = False
        self.prevWaitingVehicles = {}
        self.preWaitingSort = []
        return obs
    
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
        split_allActions = os.path.splitext(allActions)
        new_allActions = split_allActions[0] + '_' + str(self.resetCount//self.resetInterval) + split_allActions[1]
        # while os.path.exists(os.path.join(dest_folder, new_allActions)):
        #     counter += 1
        #     new_allActions = split_allActions[0] + '_' + str(counter) + split_allActions[1]
        dest_path = os.path.join(dest_folder, new_allActions)
        with open(dest_path, 'w') as f:
            # write some text to the file
            f.write(self.allActionSpace)
        # self.allActionSpace = ""
        # return counter

    def create_new_replay(self):
        # self.engine.set_save_replay(False) # stop saving replay
        # src_folder = self.configDict["dir"].replace("/", "")
        dest_folder = self.outputFolder 

        replayFilename = self.configDict["replayLogFile"]
        split_replayFilename = os.path.splitext(replayFilename)
        replayFilename = split_replayFilename[0][:-1]
        new_replayFilename = replayFilename + str(self.resetCount//self.resetInterval) + split_replayFilename[1] 
        # src_path = os.path.join(src_folder, replayFilename)
        dest_path = os.path.join(dest_folder, new_replayFilename)
        # # # shutil.move(src_path, dest_path) # copy for replayLogFile
        # dont need write with open(dest_path, 'w') as f:
        #     pass
        self.engine.set_replay_file(new_replayFilename) # set replay file back to source file to create txt file
        print("Replay file set to ", dest_path)
        # self.engine.set_save_replay(True) # start saving replay

        print("Count: ", self.resetCount+1)
        print("Replay, roadnet log files, and allAction file have been moved to ", dest_folder)


    def render(self, mode='human', close=False):
        pass