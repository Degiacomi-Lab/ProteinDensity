import numpy as np
import itertools
import MDAnalysis as mda
from MDAnalysis import transformations
import scipy.spatial.distance as S
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from functions.find_internal_waters import find_internal_waters
from functions.sasa import sasa
        
ATOMIC_RADII = {'H'   : 0.120, 'He'  : 0.140, 'Li'  : 0.076, 'Be' : 0.059,
                'B'   : 0.192, 'C'   : 0.170, 'N'   : 0.155, 'O'  : 0.152,
                'F'   : 0.147, 'Ne'  : 0.154, 'Na'  : 0.102, 'Mg' : 0.086,
                'Al'  : 0.184, 'Si'  : 0.210, 'P'   : 0.180, 'S'  : 0.180,   
                'Cl'  : 0.181, 'Ar'  : 0.188, 'K'   : 0.138, 'Ca' : 0.114,
                'Sc'  : 0.211, 'Ti'  : 0.200, 'V'   : 0.200, 'Cr' : 0.200,     
                'Mn'  : 0.200, 'Fe'  : 0.200, 'Co'  : 0.200, 'Ni' : 0.163,     
                'Cu'  : 0.140, 'Zn'  : 0.139, 'Ga'  : 0.187, 'Ge' : 0.211,        
                'As'  : 0.185, 'Se'  : 0.190, 'Br'  : 0.185, 'Kr' : 0.202,      
                'Rb'  : 0.303, 'Sr'  : 0.249, 'Y'   : 0.200, 'Zr' : 0.200,     
                'Nb'  : 0.200, 'Mo'  : 0.200, 'Tc'  : 0.200, 'Ru' : 0.200,      
                'Rh'  : 0.200, 'Pd'  : 0.163, 'Ag'  : 0.172, 'Cd' : 0.158,     
                'In'  : 0.193, 'Sn'  : 0.217, 'Sb'  : 0.206, 'Te' : 0.206,     
                'I'   : 0.198, 'Xe'  : 0.216, 'Cs'  : 0.167, 'Ba' : 0.149,       
                'La'  : 0.200, 'Ce'  : 0.200, 'Pr'  : 0.200, 'Nd' : 0.200,    
                'Pm'  : 0.200, 'Sm'  : 0.200, 'Eu'  : 0.200, 'Gd' : 0.200,      
                'Tb'  : 0.200, 'Dy'  : 0.200, 'Ho'  : 0.200, 'Er' : 0.200,       
                'Tm'  : 0.200, 'Yb'  : 0.200, 'Lu'  : 0.200, 'Hf' : 0.200,     
                'Ta'  : 0.200, 'W'   : 0.200, 'Re'  : 0.200, 'Os' : 0.200,      
                'Ir'  : 0.200, 'Pt'  : 0.175, 'Au'  : 0.166, 'Hg' : 0.155,      
                'Tl'  : 0.196, 'Pb'  : 0.202, 'Bi'  : 0.207, 'Po' : 0.197,       
                'At'  : 0.202, 'Rn'  : 0.220, 'Fr'  : 0.348, 'Ra' : 0.283,      
                'Ac'  : 0.200, 'Th'  : 0.200, 'Pa'  : 0.200, 'U'  : 0.186,     
                'Np'  : 0.200, 'Pu'  : 0.200, 'Am'  : 0.200, 'Cm' : 0.200,      
                'Bk'  : 0.200, 'Cf'  : 0.200, 'Es'  : 0.200, 'Fm' : 0.200,     
                'Md'  : 0.200, 'No'  : 0.200, 'Lr'  : 0.200, 'Rf' : 0.200,      
                'Db'  : 0.200, 'Sg'  : 0.200, 'Bh'  : 0.200, 'Hs' : 0.200,     
                'Mt'  : 0.200, 'Ds'  : 0.200, 'Rg'  : 0.200, 'Cn' : 0.200,      
                'Uut' : 0.200, 'Fl'  : 0.200, 'Uup' : 0.200, 'Lv' : 0.200,      
                'Uus' : 0.200, 'Uuo' : 0.200} 


def protein_volume_void_box(pdb, step=1, leeway=5, shell=6, rotate = False, surface_atoms=[], sasa_theshold=0.05, void_threshold=3, box=True, boxnum = 5, box_leeway=5):
    
    #PDB has to be a string pointing to a pdb file (for now...)
    U = mda.Universe(pdb)
    protein = U.select_atoms("protein")
    solvent = U.select_atoms("resname SOL or resname WAT")

    #get vdw radii for protein and water
    prot_elements = protein.atoms.elements
    prot_radii = [ATOMIC_RADII[element]*10 for element in prot_elements]
    
    wat_elements = solvent.atoms.elements
    wat_radii = [ATOMIC_RADII[element]*10 for element in wat_elements]
    
    
    if rotate == True:
        angle = np.random.random(1)*360
        direction = np.random.default_rng().uniform(-1,1, (1,3))
        ts = U.trajectory.ts
        ag = U.atoms
        rotated = transformations.rotate.rotateby(angle, direction, ag=ag)(ts)

    
    if len(surface_atoms) == 0:
        _,_,surface_atoms,_ = sasa(pdb, threshold = sasa_theshold)
    
    surface_atoms = set(surface_atoms)
    
    prot_positions = protein.positions
    sol_positions = solvent.positions
    
    minpos = np.min(prot_positions, axis = 0) - leeway #How low could this distance go?
    maxpos = np.max(prot_positions, axis = 0) + leeway
    myxrange = np.arange(minpos[0], maxpos[0], step)
    myyrange = np.arange(minpos[1], maxpos[1], step)
    myzrange = np.arange(minpos[2], maxpos[2], step)
    
    if box == False:
        cnt = 0
        void_cnt = 0
        pts = []
        void_pts = []
    
        for m in myzrange:
            #itertools.products finds the set of all ordered products between x, y, and z ranges. 
            #I.e. the below gives a segment of the full 3D grid for a specific z value.
            #All points in this 'slice' are loaded as temporary points
            pts_tmp = np.array(list(itertools.product(myxrange, myyrange, np.array([m]))))
            
            #Find the distances between each point in slice and each protein and water atoms
            p_dists = S.cdist(pts_tmp, prot_positions)
            w_dists = S.cdist(pts_tmp, sol_positions)
            
            #For each point in each slice the minimum distance to the nearest protein and water atoms are found
            dist_p = np.min(p_dists, axis=1)
            dist_w = np.min(w_dists, axis=1)
            
            cnt += np.sum(dist_p<=dist_w)
            
            #Identify 'void' voxels
            
            #Find the indices of the closest protein and water atoms to each grid point
            p_close = np.argmin(p_dists, axis=1)
            w_close = np.argmin(w_dists, axis=1)
            
            #check if nearest protein atoms are surface atoms
            p_close_surf = np.array([True if prot_atom in surface_atoms else False for prot_atom in p_close])
            
            #find radii of nearest atoms
            p_close_radii = [prot_radii[i] for i in p_close]
            w_close_radii = [wat_radii[i] for i in w_close]
            
            # outside vdw condition
            vdw_cond = np.logical_and(dist_p > p_close_radii, dist_w > w_close_radii)
            
            # Close to water and nearest protein surface condition
            surf_cond = np.logical_and(dist_w < void_threshold, p_close_surf == True)
            
            # Must be close to protein too
            surf_cond = np.logical_and(surf_cond, dist_p < void_threshold)
            
            # Think there is too much void between water atoms to be able to accurately 
            # count the whole void volume - it's difficult to define where the void
            # should begin and end.
            # Instead, add another condition to only condsider the void that overlaps
            # with volume defined as protein
            
            surf_cond = np.logical_and(surf_cond, dist_p <= dist_w)
            
            # Identify void based on 
            void = np.where(np.logical_and(vdw_cond, surf_cond))[0]
            
            if len(void) > 0:
                void_indices = np.argwhere(np.logical_and(vdw_cond, surf_cond)) 
                void_pts_tmp = np.array([pts_tmp[i][0] for i in void_indices])
                void_pts.append(void_pts_tmp)
                void_cnt += len(void)        
    
            pts.extend(pts_tmp[np.abs(dist_p - dist_w)< step])
            
        vol = cnt*(step**3)
        void_vol = void_cnt*(step**3)
        
        return vol, void_vol, np.vstack(void_pts)
    
    else:
        
        # implement double cubic lattice like structure by dividing the grid into boxes
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
        
        box_ranges = list(map(boxsplit, [myxrange, myyrange, myzrange]))
        box_indices = itertools.product(range(boxnum), range(boxnum), range(boxnum))
        
        # ensure boxnums is less than the number of grid lines in each dimension
        
        assert boxnum <= len(myxrange), "The number of boxes is too high for the given step size. Please either choose less boxes or a smaller step size."
        assert boxnum <= len(myyrange), "The number of boxes is too high for the given step size. Please either choose less boxes or a smaller step size."
        assert boxnum <= len(myzrange), "The number of boxes is too high for the given step size. Please either choose less boxes or a smaller step size."

        
        # count of number of voxels to be included in the protein
        cnt = 0
        # count number of void voxels
        void_cnt = 0
        
        # points at the centre of each voxel determined to be part of the protein
        pts = []
        # points at the centre of each voxel determined to be part of the void
        void_pts = []

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
                
                # Get the original indices of the atoms within the current box
                box_prot_indices = np.where(
                    (prot_positions[:, 0] >= min(box_xrange)-box_leeway) & (prot_positions[:, 0] <= max(box_xrange)+box_leeway) &
                    (prot_positions[:, 1] >= min(box_yrange)-box_leeway) & (prot_positions[:, 1] <= max(box_yrange)+box_leeway) &
                    (prot_positions[:, 2] >= min(box_zrange)-box_leeway) & (prot_positions[:, 2] <= max(box_zrange)+box_leeway)
                )[0]
                
                box_sol_positions = sol_positions[
                    (sol_positions[:, 0] >= min(box_xrange)-box_leeway) & (sol_positions[:, 0] <= max(box_xrange)+box_leeway) &
                    (sol_positions[:, 1] >= min(box_yrange)-box_leeway) & (sol_positions[:, 1] <= max(box_yrange)+box_leeway) &
                    (sol_positions[:, 2] >= min(box_zrange)-box_leeway) & (sol_positions[:, 2] <= max(box_zrange)+box_leeway)
                ]
                
                box_sol_indices = np.where(
                    (sol_positions[:, 0] >= min(box_xrange)-box_leeway) & (sol_positions[:, 0] <= max(box_xrange)+box_leeway) &
                    (sol_positions[:, 1] >= min(box_yrange)-box_leeway) & (sol_positions[:, 1] <= max(box_yrange)+box_leeway) &
                    (sol_positions[:, 2] >= min(box_zrange)-box_leeway) & (sol_positions[:, 2] <= max(box_zrange)+box_leeway)
                )[0]
                
                #distance from voxel to all protein and water atoms in box + leeway
                p_dists = S.cdist(pts_tmp, box_prot_positions)
                w_dists = S.cdist(pts_tmp, box_sol_positions)
                
                #For each point in each slice the minimum distance to the nearest protein and water atoms are found
                # if there are no protein or water atoms in box, skip box
                
                # Actually, box could contain only protein and no water, wouldn't want to skip...
                
                #Find the distances and indices of the closest protein and water atoms to each grid point

                try:
                    dist_p = np.min(p_dists, axis=1)
                    p_close = np.argmin(p_dists, axis=1)

                except:
                    # if box contains no protein atoms then it contains no protein, so skip box
                    # but could it contain void volume? Not as void volume is currently defined, as selecting the void
                    # that is otherwise classed as part of the protein
                    continue
                
                try:
                    dist_w = np.min(w_dists, axis=1)
                    w_close = np.argmin(w_dists, axis=1)

                except:
                    #in this case, the box contains protein but not water, so contains no void volume
                    continue

                #include all the voxels of the slice for which distance to protein is 
                #less than distance to water in the protein volume
                cnt += np.sum(dist_p<=dist_w)
                pts.extend(pts_tmp[np.abs(dist_p - dist_w) < step])
                
                #Identify 'void' voxels
                
                # Find the indices of the closest protein and water atoms (with respect to the original indices)
                p_close_original_indices = box_prot_indices[p_close]
                w_close_original_indices = box_sol_indices[w_close]
                
                # Check if the closest protein atoms are surface atoms
                p_close_surf = np.array([True if prot_atom in surface_atoms else False for prot_atom in p_close_original_indices])
                
                #find radii of nearest atoms
                p_close_radii = [prot_radii[i] for i in p_close_original_indices]
                w_close_radii = [wat_radii[i] for i in w_close_original_indices]
                
                # outside vdw condition
                vdw_cond = np.logical_and(dist_p > p_close_radii, dist_w > w_close_radii)
                
                # Close to water and nearest protein surface condition
                surf_cond = np.logical_and(dist_w < void_threshold, p_close_surf == True)
                
                # Must be close to protein too
                surf_cond = np.logical_and(surf_cond, dist_p < void_threshold)
                
                # Think there is too much void between water atoms to be able to accurately 
                # count the whole void volume - it's difficult to define where the void
                # should begin and end.
                # Instead, add another condition to only condsider the void that overlaps
                # with volume defined as protein
                
                surf_cond = np.logical_and(surf_cond, dist_p <= dist_w)
                
                # Identify void based on 
                void = np.where(np.logical_and(vdw_cond, surf_cond))[0]
                
                if len(void) > 0:
                    void_indices = np.argwhere(np.logical_and(vdw_cond, surf_cond)) 
                    void_pts_tmp = np.array([pts_tmp[i][0] for i in void_indices])
                    void_pts.append(void_pts_tmp)
                    void_cnt += len(void)        
        
                
        vol = cnt*(step**3)
        void_vol = void_cnt*(step**3)
        
        if len(void_pts) > 0:
            void_pts = np.vstack(void_pts)
            
        return vol, void_vol, void_pts
    
    
    
