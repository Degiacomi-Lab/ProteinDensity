# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:34:35 2024

@author: hkwf34
"""

#%% Example usage
import Classes as Cs
import pickle
from itertools import islice


all_structure_results = {}
all_seq_results = {}


#%%
with open("C:/Users/hkwf34/OneDrive - Durham University/Desktop/proteins/rf_results/all_structure_results_prelim.pickle", "rb") as r_file:
    all_structure_results = pickle.load(r_file)
#%%
problems = []

for i, file in enumerate(unmatched_files):
    
    pdb = file.split("\\")[-1].split("_sim")[0]
    
    if pdb in list(all_structure_results.keys()):
        print(f"{pdb} done")
        continue

    print(f"Trying {pdb}...")
    try:
        
        short_file = f"{file.split(".")[0]}_short.pdb"
        with open(file, 'r') as full_structure, open(short_file, 'w') as short_structure:
            short_structure.writelines(islice(full_structure, 60000))
    
        struct = Cs.Protein(short_file, "structure") 
        
        sequence = haddocking_sequences[pdb]
        
        seq = Cs.Protein(sequence, "sequence") 
        
        seq_results = seq.predict()
        
        struct_results = struct.predict()
        
        all_structure_results[pdb] = struct_results
        
        all_seq_results[pdb] = seq_results
        
    except:
        print(f"Issue for\n{file}")
        problems.append(file)
        continue
    
    
    print(i)

#%%
with open("random_forests/RF_300_means.pickle", "rb") as r_file:
    RF_structure_means = pickle.load(r_file)
        
struct_results_means = struct.predict(RF_structure_means)
