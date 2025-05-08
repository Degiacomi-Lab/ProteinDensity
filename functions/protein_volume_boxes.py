import numpy as np
import itertools
import MDAnalysis as mda
from MDAnalysis import transformations
import scipy.spatial.distance as S

def protein_volume_boxes(pdb, step=1, leeway=5, shell=6, boxnum=8, box_leeway=5, rotate=False):
    
    #if pdb is a string then assume it is a string that refers to a pdb file in
    #the cwd. Create a MDA Universe of this pdb. Select protein atoms.
    if type(pdb) is str:
        U = mda.Universe(pdb)
        protein = U.select_atoms("protein")
                
    #if pdb file is not a string, assume it is already an MDA Universe or atom group
    #Select protein atoms.
    else:
        U = pdb
        protein = U.select_atoms("protein")
        
    if rotate == True:
        angle = np.random.random(1)*360
        direction = np.random.default_rng().uniform(-1,1, (1,3))
        ts = U.trajectory.ts
        ag = U.atoms
        rotated = transformations.rotate.rotateby(angle, direction, ag=ag)(ts)

    # define the solvent atoms and get protein and solvent cartesian coordinate positions
    solvent = U.select_atoms(f"(resname SOL or resname WAT) and around {shell} protein")
    prot_positions = protein.positions
    sol_positions = solvent.positions
    
    # define the entire grid that contains the protein, based on the maximum
    # extent of the protein in cartesian axes, the leeway, and the step size
    minpos = np.min(prot_positions, axis = 0) - leeway
    maxpos = np.max(prot_positions, axis = 0) + leeway
    xrange = np.arange(minpos[0], maxpos[0], step)
    yrange = np.arange(minpos[1], maxpos[1], step)
    zrange = np.arange(minpos[2], maxpos[2], step)
    
    # implement memory-saving boxes by dividing the grid into several component grids
    # defined by a multiple of the step size, boxsize
        
    def boxsplit(arr, n=boxnum):
        # Calculate the length of each part
        avg_length = len(arr) / n
        box_ranges = []
        last_index = 0.0
    
        for i in range(n):
            # Calculate the start and end indices for the current part
            start_index = int(last_index)
            last_index += avg_length
            end_index = int(last_index)
            
            # Slice the array to get the current part
            box_ranges.append(arr[start_index:end_index])
    
        return box_ranges
    
    box_ranges = list(map(boxsplit, [xrange, yrange, zrange]))
    box_indices = itertools.product(range(boxnum), range(boxnum), range(boxnum))
    
    # ensure boxnums is less than the number of grid lines in each dimension
    
    assert boxnum <= len(xrange), "The number of boxes is too high for the given step size. Please either choose less boxes or a smaller step size."
    assert boxnum <= len(yrange), "The number of boxes is too high for the given step size. Please either choose less boxes or a smaller step size."
    assert boxnum <= len(zrange), "The number of boxes is too high for the given step size. Please either choose less boxes or a smaller step size."

    
    # count of number of voxels to be included in the protein
    cnt = 0
      
    # points at the centre of each voxel determined to be part of the protein
    pts = []
    
    for box_index in box_indices:
        box_xrange, box_yrange, box_zrange = box_ranges[0][box_index[0]], box_ranges[1][box_index[1]], box_ranges[2][box_index[2]]
        for m in box_zrange:
            #itertools.products finds the set of all ordered products between x, y, and z ranges. 
            #I.e. the below gives a segment of the full 3D grid for a specific z value.
            #All points in this 'slice' are loaded as temporary points
            pts_tmp = np.array(list(itertools.product(box_xrange, box_yrange, np.array([m]))))

            box_prot_positions = prot_positions[
                (prot_positions[:, 0] >= min(box_xrange)-box_leeway) & (prot_positions[:, 0] <= max(box_xrange)+box_leeway) &
                (prot_positions[:, 1] >= min(box_yrange)-box_leeway) & (prot_positions[:, 1] <= max(box_yrange)+box_leeway) &
                (prot_positions[:, 2] >= min(box_zrange)-box_leeway) & (prot_positions[:, 2] <= max(box_zrange)+box_leeway)
            ]

            box_sol_positions = sol_positions[
                (sol_positions[:, 0] >= min(box_xrange)-box_leeway) & (sol_positions[:, 0] <= max(box_xrange)+box_leeway) &
                (sol_positions[:, 1] >= min(box_yrange)-box_leeway) & (sol_positions[:, 1] <= max(box_yrange)+box_leeway) &
                (sol_positions[:, 2] >= min(box_zrange)-box_leeway) & (sol_positions[:, 2] <= max(box_zrange)+box_leeway)
            ]
            #For each point in each slice the minimum distance to the nearest protein and water atoms are found
            try:
                dist_p = np.min(S.cdist(pts_tmp, box_prot_positions), axis=1)
            except:
                continue
            
            try:
                dist_w = np.min(S.cdist(pts_tmp, box_sol_positions), axis=1)

            except:
                dist_w = np.ones(np.shape(dist_p)) * np.inf
                continue
            
                    
            #include all the voxels of the slice for which distance to protein is 
            #less than distance to water in the protein volume
            cnt += np.sum(dist_p<=dist_w)
            pts.extend(pts_tmp[np.abs(dist_p - dist_w)< step])
            
    vol = cnt*(step**3)
            
    return vol, np.array(pts)
