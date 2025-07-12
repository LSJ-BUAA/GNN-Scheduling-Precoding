# GNN for Wideband  User Scheduling and Hybrid Precoding

## Introduction

This repository includes the simulation code of the following paper.

>  Shengjie Liu, Chenyang Yang, and Shengqian Han, "Learning Wideband User Scheduling and Hybrid Precoding with Graph Neural Networks," *IEEE Transactions on Wireless Communications*, accepted, 2025.

## Usage

- Generate datasets:
	- Generate training and testing channel datasets using `data/genChannel.m`.
- Train and test models:
	- Pre-train the **precoder module** using `networks/precoder_main.py`.
	- Jointly train the **scheduler module** and the **precoder module** using `networks/scheduler_main.py`.
	- Test results are reported after each training epoch.
- Evaluate size generalizability:
	- Use the files in folder `networks/generalization` to evaluate the size generalization performance across different system scales. 
