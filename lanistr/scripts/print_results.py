import torch
import sys  
import pprint  

path = sys.argv[1]

results = torch.load(path)

pprint.pprint(results)
