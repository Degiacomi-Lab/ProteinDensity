import numpy as np
import itertools
import MDAnalysis as mda
from MDAnalysis import transformations
import scipy.spatial.distance as S
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from functions.find_internal_waters import find_internal_waters
from functions.sasa_residues import sasa
import pandas as pan
       
def protein_volume_residues(pdb, step=0.5, leeway=5, threshold = 6, surf_threshold_percent = 5, rotate = False, skip_tol = 0):

    #if pdb is a string then assume it is a string that refers to a pdb file in
    #the cwd. Create a MDA Universe of this pdb. Select protein atoms.
    if type(pdb) is str:
        U = mda.Universe(pdb)
        protein = U.select_atoms("protein")
       
    #if pdb file is not a string, assume it is already an MDA Universe
    #Select protein atoms.
    elif type(pdb) == mda.core.universe.Universe:
        U = pdb
        protein = U.select_atoms("protein")
    
    else:
        print("Error: PDB is neither a protein structure file or an MDAnalysis Universe object.")
        return
    
    if rotate == True:
        angle = np.random.random(1)*360
        direction = np.random.default_rng().uniform(-1,1, (1,3))
        ts = U.trajectory.ts
        ag = U.atoms
        rotated = transformations.rotate.rotateby(angle, direction, ag=ag)(ts)

    asa, mesh_pts, surface_atoms, indices, cnts = sasa(pdb, threshold = 0, return_count = True)
    
    atom_resnums = protein.resnums
       
    cnts_percent = 100*(cnts/960)

    # Find positions of protein atoms
    protein_positions = protein.positions
    
    # Build a cuboid box of voxels around the entire protein
    minpos = np.min(protein_positions, axis = 0) - leeway 
    maxpos = np.max(protein_positions, axis = 0) + leeway
    myxrange = np.arange(minpos[0], maxpos[0], step)
    myyrange = np.arange(minpos[1], maxpos[1], step)
    myzrange = np.arange(minpos[2], maxpos[2], step)
    
    # Initiate total count list, so we obtain volumes per residue
    total_cnt = []

    resnum = 1 # MDAnalysis residue numbering starts from 1
    
    # Array of all voxels
    all_voxels = np.zeros((len(myxrange),len(myyrange),len(myzrange)))
    
    warning = 0
    
    resnums = []
    while True:
        ag = U.select_atoms(f"protein and resnum {resnum}")
        if warning > skip_tol:
            print(f"Breaking for residue {resnum}")
            break

        if len(ag) == 0:  # No atoms selected, stop the loop
            warning += 1
            resnum += 1
            continue
      
        else:

            resnums.append(resnum)
            # Find positions of protein atoms in residue
            residue_positions = ag.positions
            
            # Find positions of all other relevant atoms, then their positions
            other = U.select_atoms(f"around {threshold} resnum {resnum}")
            other_positions = other.positions
            
            # Rather than building a new grid box for each residue, assign 
            # voxels from the intial grid covering the whole protein to each residue

            residue_minpos = np.min(residue_positions, axis = 0) - leeway 
            residue_maxpos = np.max(residue_positions, axis = 0) + leeway
            
            
            residue_xrange = myxrange[np.logical_and(myxrange>=residue_minpos[0], myxrange<=residue_maxpos[0])]
            residue_yrange = myyrange[np.logical_and(myyrange>=residue_minpos[1], myyrange<=residue_maxpos[1])]
            residue_zrange = myzrange[np.logical_and(myzrange>=residue_minpos[2], myzrange<=residue_maxpos[2])]
            
            
            pts = []
            
            cnt = 0
            for m in residue_zrange:
                #itertools.products finds the set of all ordered products between x, y, and z ranges. 
                #I.e. the below gives a segment of the full 3D grid for a specific z value.
                #All points in this 'slice' are loaded as temporary points
                pts_tmp = np.array(list(itertools.product(residue_xrange, residue_yrange, np.array([m]))))
                
                #For each point in each slice the minimum distance to the nearest residue and other atoms are found
                dist_r = np.min(S.cdist(pts_tmp, residue_positions), axis=1)
                
                #It is possible (especially if the threshold is set too low) for there
                #to be no 'other' atoms. In this case, the whole box constructed around 
                #the protein would have to be considered part of the protein
                try:
                    dist_o = np.min(S.cdist(pts_tmp, other_positions), axis=1)
                except:
                    print(f"Warning: no other atoms surrounding residue {resnum}.\nYour theshold value ({threshold}) may be too low.")
                    cnt += len(pts_tmp)
                    pts.extend(pts_tmp)
                    continue
                
                #include all the voxels of the slice for which distance to residue atom is 
                #less than distance to other atoms in the protein volume
                cnt += np.sum(dist_r<=dist_o)
                pts.extend(pts_tmp[np.abs(dist_r - dist_o)< step])
                
            # Add the count for each residue to the total count
            total_cnt.append(cnt)
            
            resnum += 1  # Increment residue number 

    residue_vols = np.array(total_cnt)*(step**3)
    vol = sum(total_cnt)*(step**3)
    
    residue_exposed_percents = np.array([np.mean(cnts_percent[np.where(atom_resnums==resnum)[0]]) for resnum in resnums])

    #return protein.residues.resnames, residue_vols, residue_exposed_percents, resnums

    data = pan.DataFrame()
    data["Residues"] = protein.residues.resnames
    data["Residue volume"] = residue_vols
    data["Solvent exposure / %"] = residue_exposed_percents
    data["Residue location"] = ["surface" if percent >= surf_threshold_percent else "interior" for percent in residue_exposed_percents]
    data["File"] = pdb.split("/")[1]
    data["Total vol"] = vol

    return vol, data
