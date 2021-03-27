"""
Created on 2020/1/14

@author: Boyun Li
"""

import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--name', type=str, default="SOTS_GT")
# parser.add_argument('--datasets', type=str, default="SOTS")
parser.add_argument('--datasets', type=str, default="HSTS")
# parser.add_argument('--datasets', type=str, default="real-world")

parser.add_argument('--clip', type=bool, default=True)
parser.add_argument('--num_iter', type=int, default=800)
parser.add_argument('--learning_rate', type=float, default=0.001)

options = parser.parse_args()
