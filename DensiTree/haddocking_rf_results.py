# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:34:35 2024

@author: hkwf34
"""

#%% Example usage
import Classes as Cs
import pickle
from itertools import islice
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pan

#%%

all_structure_results = {}
all_seq_results = {}

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
        
#%%

struct_results = [val["Structure prediction"] for val in all_structure_results.values()]
seq_results = [val["Sequence prediction"] for val in all_seq_results.values()]


#%%

plt.scatter(range(60), struct_results, color="red")

plt.scatter(range(60), seq_results, color="green")

#%%


sns.kdeplot(struct_results, color="red")

sns.kdeplot(seq_results, color="green")

#%% import results from haddocking density calculations



#%% Import random forests trained on 201-protein dataset
import pickle 

with open("RF_300.pickle", "rb") as r_file:
    RF_300 = pickle.load(r_file)

with open("RF_300_means.pickle", "rb") as r_file:
    RF_300_means = pickle.load(r_file)

with open("RF_300_seq.pickle", "rb") as r_file:
    RF_300_seq = pickle.load(r_file)
    
#%% Import 201-protein dataset 300k data

df300 = pan.read_csv("300k_repeat/300k_structure_feats.csv")
df300_means = pan.read_csv("300k_repeat/300k_structure_feats_means.csv")
df300_seq = pan.read_csv("300k_repeat/300k_sequence_feats.csv")
    
dens300 = df300["densities"]
dens300_means = df300_means["densities"]
dens300_seq = df300_seq["densities"]

feats300 = df300.drop(columns=["densities"])
feats300_means = df300_means.drop(columns=["densities"])
feats300_seq = df300_seq.drop(columns=["densities"])

#%%


sns.kdeplot(struct_results, color="red", label="Test set\nstructure RF")

sns.kdeplot(seq_results, color="green", label="Test set\nsequence RF")

sns.kdeplot(dens300, color="blue", label="201-protein dataset")

plt.ylabel("Frequency density")
plt.xlabel("Protein mass density / g cm$^{-3}$")
plt.legend()

#%% Save haddocking results for 0 ns simulation frames

import pickle

with open("c:/users/hkwf34/onedrive - durham university/desktop/proteins/haddocking/structure_results_frame0.pickle", "wb") as w_file:
    pickle.dump(all_structure_results, w_file)
    
#%% LOAD
import pickle

with open("c:/users/hkwf34/onedrive - durham university/desktop/proteins/haddocking/structure_results_frame0.pickle", "rb") as r_file:
    all_structure_results = pickle.load(r_file)

#%% Repeat for all 21 files for each Haddocking structure

all_structure_results_21 = {f"{key}_0":val for key, val in all_structure_results.items()}

all_seq_results_21 =  {f"{key}_0":val for key, val in all_seq_results.items()}

#%%


unmatched_files_21 = unmatched_files.copy()

for file in unmatched_files:
    new_files = [file.replace("_sim0", f"_sim{i}") for i in range(1, 5)]
    unmatched_files_21 += new_files
    
#%%
with open("rf_results/haddocking_sequences_unique.pickle", "rb") as pickle_file:
    haddocking_sequences_unique = pickle.load(pickle_file)

#%%
import os
from pathlib import Path

source_folder = Path("P:/hkwf34/EARS/haddocking/sim_pdbs")

unmatched_files  = [os.path.join(source_folder, f"{pdb}_sim0.pdb") for pdb in list(haddocking_sequences_unique.keys())]

#%%
unmatched_files_21 = unmatched_files.copy()

for file in unmatched_files:
    new_files = [file.replace("_sim0", f"_sim{i}") for i in range(1, 5)]
    unmatched_files_21 += new_files
    


#%%

import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os

copy_files = [
    file
    for file in unmatched_files_21
    if f"{file.split('\\')[-1].split('_sim')[0]}_{file.split('\\')[-1].split('_sim')[1].split('.')[0]}"
    not in all_structure_results_21.keys()
]


source_folder = Path("P:/hkwf34/EARS/haddocking/sim_pdbs")

destination_folder = Path("C:/users/hkwf34/onedrive - durham university/desktop/proteins/haddocking/pdb_files")
destination_folder.mkdir(parents=True, exist_ok=True)

def copy_file(file_path):
    print(f"Starting file {file_path}")
    destination = destination_folder / file_path.split("\\")[-1]
    shutil.copy2(file_path, destination)
    print(f"Finished file {file_path}")

copied = os.listdir(destination_folder)

file_paths = [f for f in copy_files if f.split("\\")[-1] not in copied]

#%%
print("Getting started...")
# Use ThreadPoolExecutor for I/O-bound operations
with ThreadPoolExecutor() as executor:
    executor.map(copy_file, file_paths)

#%% save preliminary results


with open("c:/users/hkwf34/onedrive - durham university/desktop/proteins/rf_results/all_seq_results.pickle", "wb") as w_file:
    pickle.dump(all_seq_results, w_file)
    
with open("c:/users/hkwf34/onedrive - durham university/desktop/proteins/rf_results/all_structure_results_prelim.pickle", "wb") as w_file:
    pickle.dump(all_structure_results_21, w_file)

#%% load preliminary results

with open("c:/users/hkwf34/onedrive - durham university/desktop/proteins/rf_results/all_structure_results_prelim.pickle", "rb") as r_file:
    all_structure_results_21 = pickle.load(r_file)

#%%
import time
import MDAnalysis as mda
import numpy as np

copied_dest = [os.path.join(destination_folder, copy) for copy in copied if np.logical_and("short" not in copy, "water" not in copy)]
#%%

all_sequence_results = {}
problems = []

for i, file in enumerate(copied_dest):
    

    pdb_code = file.split("\\")[-1].split("_sim")[0]
    
    frame = file.split("\\")[-1].split("_sim")[1].split(".")[0]
    
    pdb = f"{pdb_code}_{frame}"
    
    if pdb in list(all_structure_results_21.keys()):
        print(f"{pdb} done")
        continue

    print(f"Trying {pdb}...")
    
    start =  time.perf_counter()

    try:
            
        protein = mda.Universe(file).select_atoms("protein")
        
        nowater_file = file.replace("_sim", "_sim_nowater")
        protein.write(nowater_file)
        
        
        struct = Cs.Protein(nowater_file, "structure") 
        

        
        struct_results = struct.predict()
        
        all_structure_results_21[pdb] = struct_results
        
        if pdb not in all_sequence_results.keys():
            seq_pdb = "_".join(pdb.split("_")[:-1])

            sequence = haddocking_sequences_unique[seq_pdb]
            
            seq = Cs.Protein(sequence, "sequence") 
            
            seq_results = seq.predict()
            
            all_sequence_results[pdb] = seq_results
        
        end = time.perf_counter()
        
        print(end-start)

    
    except:
        
        try:
            
            short_file = f"{file.split(".")[0]}_short.pdb"
            with open(file, 'r') as full_structure, open(short_file, 'w') as short_structure:
                short_structure.writelines(islice(full_structure, 60000))
        
        
            struct = Cs.Protein(short_file, "structure") 
            
            
            struct_results = struct.predict()
            
            all_structure_results_21[pdb] = struct_results
            
            if pdb not in all_sequence_results.keys():
                
                sequence = haddocking_sequences_unique[seq_pdb]
                
                seq = Cs.Protein(sequence, "sequence") 
                
                seq_results = seq.predict()
                
                all_sequence_results[pdb] = seq_results
            
            
        except:
            print(f"Issue for\n{file}")
            problems.append(file)
            continue
        
    
    print(i)


#%% save final results

with open("c:/users/hkwf34/onedrive - durham university/desktop/proteins/rf_results/all_structure_results_final.pickle", "wb") as w_file:
    pickle.dump(all_structure_results_21, w_file)

#%% load final results

with open("c:/users/hkwf34/onedrive - durham university/desktop/proteins/rf_results/all_structure_results_final.pickle", "rb") as r_file:
    all_structure_results_21 = pickle.load(r_file)


#%% Unique pdb names (pdbs repesenting unique sequences)

unique_pdbs = ['2I25_r_u',
 '1JMO_r_u',
 '1YVB_l_u',
 '1PPE_l_u',
 '1BVN_r_u',
 '1FC2_r_u',
 '1GHQ_r_u',
 '1JK9_r_u',
 '2NZ8_r_u',
 '1YVB_r_u',
 '1OC0_l_u',
 '1HE1_r_u',
 '1JTD_r_u',
 '1Z5Y_r_u',
 '1JIW_r_u',
 '1NW9_r_u',
 '1DFJ_r_u',
 '1CLV_r_u',
 '1OPH_r_u',
 '1S1Q_r_u',
 '1KAC_r_u',
 '1PXV_r_u',
 '1AY7_r_u',
 '1R6Q_r_u',
 '2SIC_r_u',
 '1FQ1_r_u',
 '1JTG_l_u',
 '1AK4_r_u',
 '2O8V_r_u',
 '2AJF_r_u',
 '1MAH_l_u',
 '3A4S_r_u',
 '1WQ1_r_u',
 '1H9D_r_u',
 '1RKE_r_u',
 '1EER_r_u',
 '1KTZ_r_u',
 '1QA9_r_u',
 '1DFJ_l_u',
 '2B42_l_u',
 '2FD6_r_u',
 '1IRA_r_u',
 '2BTF_l_u',
 '1ZHH_r_u',
 '1FFW_r_u',
 '1FLE_r_u',
 '1EAW_r_u',
 '2HRK_r_u',
 '1FQJ_r_u',
 '1ACB_r_u',
 '1UDI_r_u',
 '1F34_r_u',
 '1OC0_r_u',
 '2X9A_r_u',
 '1MQ8_r_u',
 '1Y64_r_u',
 '1B6C_r_u',
 '1GPW_r_u',
 '1HIA_r_u']

with open("c:/users/hkwf34/onedrive - durham university/desktop/proteins/haddocking/unique_haddocking_pdbs.pickle", "wb") as w_file:
    pickle.dump(unique_pdbs, w_file)

#%% load unique pdb codes from haddocking dataset

with open("c:/users/hkwf34/onedrive - durham university/desktop/proteins/haddocking/unique_haddocking_pdbs.pickle", "rb") as r_file:
    unique_pdbs = pickle.load(r_file)
    
#%% Get combined dataframe with all data

# Also removes 2VXT

pdbs = []
for i, keyval in enumerate(all_structure_results_21.items()):
    key, val = keyval
    df = val["Structure feats"]
    if i == 0:
        pdb = "_".join(key.split("_")[:2])
        if f"{pdb}_u" not in unique_pdbs:
            print(pdb)
            continue
        pdbs.append(pdb)
        df["pdb"] = pdb
        all_structure_results_df = df

    else:
        pdb = "_".join(key.split("_")[:2])
        if f"{pdb}_u" not in unique_pdbs:
            print(pdb)
            continue
        pdbs.append(pdb)

        df["pdb"] = pdb
        all_structure_results_df = pan.concat([all_structure_results_df, df])
        
            
#%% Get mean dataframe for haddocking results, to combine with 201-protein dataset results

for i, pdb in enumerate(set(all_structure_results_df["pdb"])):

    if i == 0:
        mean_results_df = all_structure_results_df[all_structure_results_df["pdb"]==pdb].mean(axis=0, numeric_only = True).to_frame().T
        
        mean_results_df["pdb"] = pdb
    else:
        pdb_df = all_structure_results_df[all_structure_results_df["pdb"]==pdb]
        
        pdb_df_means = pdb_df.mean(axis=0, numeric_only = True).to_frame().T
        
        """
        errors = ['SASA', 'Disulfide bonds', 'Aspect ratio', 'Hydrophobic % surface',
       'Charged % surface', 'Other % surface', 'Acidic charged % surface',
       'Basic charged % surface', 'Aliph. hydrophobic % surface',
       'Arom. hydrophobic % surface', 'Aromatic % surface',
       'Aliphatic % surface', 'Hydrophobic % interior', 'Charged % interior',
       'Other % interior', 'Acidic charged % interior',
       'Basic charged % interior', 'Aliph. hydrophobic % interior',
       'Arom. hydrophobic % interior', 'Aromatic % interior',
       'Aliphatic % interior', 'Coil percent',
       'Strand percent', 'Helix percent', 'Rg']
        
        pdb_df_stds = pdb_df[errors].std(axis=0, numeric_only = True).to_frame().T
        """

        
        pdb_df_means["pdb"] = pdb
        
        mean_results_df = pan.concat([mean_results_df, pdb_df_means])

#%% save 59-protein (Haddocking) dataset feature values

with open("c:/users/hkwf34/onedrive - durham university/desktop/proteins/haddocking/haddocking_structure_feats_means.csv", "wb") as w_file:
    pickle.dump(mean_results_df, w_file)

#%% Combine 201-protein dataset and 59 protein dataset
dataset_201 = pan.read_csv("c:/users/hkwf34/onedrive - durham university/desktop/proteins/300k_repeat/300k_structure_feats_means.csv")

#%% Once feature removal calculation is complete, get mean data for 59 haddocking proteins,
# then combine this with the df300_means 201-protein dataset to get the full dataframe for
# mean results of the new 260-protein dataset.
# Then train a RF (using grid optimisation) on this dataset.
# Test the random forest by taking 10% of the 260 protein dataset as a test set,
# with test proteins chosen randomly, but the condition that half their densities 
# are above and half are below the mean calculated density.


