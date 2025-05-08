
import numpy as np
import MDAnalysis as mda
import itertools
import scipy.spatial.distance as S
from MDAnalysis import transformations




def shell_densities(u, zones, step, selection = "resname SOL"):
    
    def protein_zone_density(sim, step=1, leeway=5, zone = 3, rotate = False, 
                             boxnum=8, box_leeway=5, selection = "resname SOL or resname NA or resname CL"):
        
        #take a shell surrounding a protein and calculate the density inside that shell (i.e. the volume of the shell and the mass of protein and solvent inside it)
        
        U = sim
        
        if rotate == True:
            angle = np.random.random(1)*360
            direction = np.random.default_rng().uniform(-1,1, (1,3))
            ts = U.trajectory.ts
            ag = U.atoms
            rotated = transformations.rotate.rotateby(angle, direction, ag=ag)(ts)
        

        
        inner = U.select_atoms(f"({selection} and around {zone} protein) or protein")
        outer = U.select_atoms(f"(not around {zone} protein) or (not {selection} and around {zone} protein)")
        inner_mass = np.sum(inner.masses)
        
        inner_positions = inner.positions
        outer_positions = outer.positions
        

        
        
        minpos = np.min(outer_positions, axis = 0) - leeway
        maxpos = np.max(outer_positions, axis = 0) + leeway
        myxrange = np.arange(minpos[0], maxpos[0], step)
        myyrange = np.arange(minpos[1], maxpos[1], step)
        myzrange = np.arange(minpos[2], maxpos[2], step)
        
        
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
        
        box_ranges = list(map(boxsplit, [myxrange, myyrange, myzrange]))
        box_indices = itertools.product(range(boxnum), range(boxnum), range(boxnum))
        
        box_ranges_copy = list(map(boxsplit, [myxrange, myyrange, myzrange]))
        box_indices_copy = itertools.product(range(boxnum), range(boxnum), range(boxnum))
        
        # ensure boxnums is less than the number of grid lines in each dimension
        
        assert boxnum <= len(myxrange), "The number of boxes is too high for the given step size. Please either choose less boxes or a smaller step size."
        assert boxnum <= len(myyrange), "The number of boxes is too high for the given step size. Please either choose less boxes or a smaller step size."
        assert boxnum <= len(myzrange), "The number of boxes is too high for the given step size. Please either choose less boxes or a smaller step size."
        
        
        cnt = 0
        
        for box_index in box_indices:
            box_xrange, box_yrange, box_zrange = box_ranges[0][box_index[0]], box_ranges[1][box_index[1]], box_ranges[2][box_index[2]]
            for m in box_zrange:
                
                #itertools.products finds the set of all ordered products between x, y, and z ranges. 
                #I.e. the below gives a segment of the full 3D grid for a specific z value.
                #All points in this 'slice' are loaded as temporary points
                pts_tmp = np.array(list(itertools.product(box_xrange, box_yrange, np.array([m]))))
                
                
                box_inner_positions = inner_positions[
                    (inner_positions[:, 0] >= min(box_xrange)-box_leeway) & (inner_positions[:, 0] <= max(box_xrange)+box_leeway) &
                    (inner_positions[:, 1] >= min(box_yrange)-box_leeway) & (inner_positions[:, 1] <= max(box_yrange)+box_leeway) &
                    (inner_positions[:, 2] >= min(box_zrange)-box_leeway) & (inner_positions[:, 2] <= max(box_zrange)+box_leeway)
                ]
                
                box_outer_positions = outer_positions[
                    (outer_positions[:, 0] >= min(box_xrange)-box_leeway) & (outer_positions[:, 0] <= max(box_xrange)+box_leeway) &
                    (outer_positions[:, 1] >= min(box_yrange)-box_leeway) & (outer_positions[:, 1] <= max(box_yrange)+box_leeway) &
                    (outer_positions[:, 2] >= min(box_zrange)-box_leeway) & (outer_positions[:, 2] <= max(box_zrange)+box_leeway)
                ]
                
                
                
                #For each point in each slice the minimum distance to the nearest protein and water atoms are found
                try:
                    dist_inner_iw = np.min(S.cdist(pts_tmp, box_inner_positions), axis=1)
                except:
                    continue
                
                try:
                    dist_outer_iw = np.min(S.cdist(pts_tmp, box_outer_positions), axis=1)
                
                except:
                    dist_outer_iw = np.ones(np.shape(dist_inner_iw)) * np.inf
                    continue
                
                cnt += np.sum(dist_inner_iw<=dist_outer_iw)
        
     
        
        inner_vol = cnt*(step**3)
        density = (inner_mass/inner_vol)*1.6605402
                
        return density, inner_vol
    
    zone_pairs = [(zones[i], zones[i+1]) for i in range(len(zones)-1)]
    
    all_selection_densities = []
    
    # The initial enclosed mass at the start of the inner boundary of the first shell
    shell_encl_mass_initial = np.sum(u.select_atoms(f"({selection} and around {zones[0]} protein) or protein").masses)
    
    # The density enclosed by the inner boundary of the first shell
    shell_encl_density_initial, shell_encl_volume_initial = protein_zone_density(u, zone = 0, step=step, selection = selection)
    

    # The volume enclosed by the inner boundary of the first shell
    #shell_encl_volumes = [shell_encl_mass_initial/shell_encl_density_initial]
    
    # Initial shell enclosed volume is chosen as the selection-displaced protein volume
    # We want to consider shell volumes that are larger than this initial volume, so don't give negative densities
    # However, the shell boundaries are defined purely by the distance to the nearest protein atoms,
    # whereas this volume is defined more sophisticatedly based on the actual relative positions of selection and protein 
    # atoms. However, once we consider an atom as within the shell we calculate the new shell volume in the same 
    # way as the selection-displaced protein volume, we're not just looking at perfect shells.
    
    shell_encl_volumes = [shell_encl_volume_initial]
    

    shell_encl_masses = [np.sum(u.select_atoms(f"({selection} and around {zones[0]} protein)").masses)]
    
    # Store the selection densities within each shell in a dictionary, with the shell boundaries as keys
    shell_selection_densities = {}
    
    for j, zone in enumerate(zone_pairs):
        shell_encl_mass = np.sum(u.select_atoms(f"({selection} and around {zone[1]} protein) or protein").masses)
        
        #shell_encl_density_initial, shell_encl_volume_initial = protein_zone_density(u, zone = 0, step=step, selection = selection)

        shell_encl_density, shell_encl_volume = protein_zone_density(u, zone = zone[1],  step=step, selection = selection)
        #shell_encl_volume = shell_encl_mass/shell_encl_density
        
        
        shell_encl_mass = np.sum(u.select_atoms(f"({selection} and around {zone[1]} protein)").masses)*1.6605402
        
        # Subtract the volume of the previous shell, starting with the selection-displaced protein volume
        shell_volume = shell_encl_volume - shell_encl_volumes[j]
        
        shell_encl_volumes.append(shell_encl_volume)
        
        if shell_volume < 0:
            # This implies that the shell volume is smaller than the volume 'displaced' by the protein,
            # so we are not yet in an area which we could consider 'selection'
            continue
        
        # calculate the selection mass enclosed by the shell boundaries
        shell_selection_mass = shell_encl_mass - shell_encl_masses[j]
        shell_encl_masses.append(shell_encl_mass)
        # If the volume of the shell is equal to the selection-displaced protein volume, 
        # this implies no selection molecules are enclosed yet by the shell, so the density is 0.
        
        if shell_volume == 0:
            shell_selection_density = 0
        else:
            # calculate the density within the shell from the shell's mass and volume
            shell_selection_density = shell_selection_mass/shell_volume
        
        # If the shell volume isn't negative, we are in a part of the system we define as selection
        # Add to the shell_selection_densities the calculated selection density within the shell
        shell_selection_densities[zone] = shell_selection_density
    
    return shell_selection_densities