# SmartTrafficLight

## Introduction

This project uses a reinforcement learning algorithm (PPO) from gym to produce a Smart Traffic Light model that aims to optimise current traffic light systems. The model is able to output both the lightphase of the intersection, with the duration of the lightphase.

## Pre-requisites

OS System: Linux Environment (WSL works fine too)
Packages installed:
- Install the necessary packages found in requirement.txt as follows

  `pip install -r requirements.txt`
- Cityflow

  `sudo apt update && sudo apt install -y build-essential cmake` # installing cpp dependencies

  `git clone https://github.com/cityflow-project/CityFlow.git`
  
  `pip install .`
