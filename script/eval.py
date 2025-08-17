#!/usr/bin/env python
# -*- Coding: UTF8 -*-


import torch
from dataclasses import dataclass

@dataclass
class EvalParams:
	"""
	Evaluation parameters
	"""
	model_name : str # SBERT  
	model_dim : int
	data_directory = str



def load_train_test_set(config):
    
    file = open(os.path.join(config.file_directory, config.eval_file))
    csvreader = csv.reader(file, delimiter= ',')
    header = next(csvreader)
    org2patent = {}
    org2paper = {}
    patent2org = {}
    paper2org = {}
    test_set = {}
    for line in csvreader:
        
        if len(line) == 3:
            org = line[0]
            test_set[org] = org
            patent_list = line[1]
            paper_list = line[2]
            org2patent[org] = patent_list
            org2paper[org] = paper_list
            for i in patent_list.split(' '):
                if i in patent2org:
                    patent2org[i] = patent2org[i] + ' ' + org
                else:
                    patent2org[i] = org 

            for i in paper_list.split(' '):
                if i in paper2org:
                    paper2org[i] = paper2org[i] + ' ' + org
                else:
                    paper2org[i] = org        


    return (test_set, org2patent, org2paper, patent2org,paper2org)	




if __name__ == '__main__':
	print("-> Start evaluation... ")
	# Config information

	
	
	config = EvalParams(
		model_name = "SBERT", 
		model_dim=384,
		data_directory = "../data/",
		eval_file = "../data/dev.txt" 
		)

	file_name
	print(config)

	# Load data 
	(test_set, org2patent, org2paper, patent2org, paper2org) = load_data(config)

