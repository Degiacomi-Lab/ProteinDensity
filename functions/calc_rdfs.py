# -*- coding: utf-8 -*-

import numpy as np
import os
import MDAnalysis as mda
import matplotlib.pyplot as plt
import MDAnalysis.analysis.rdf as rdf
import scipy.spatial.distance as S
from MDAnalysis import transformations
import itertools

def protein_zone_density(pdb, step=0.5, leeway=5, zone = 3, rotate = False, internal_waters = False, internal_water_method = "positions"):
    
    #take a shell surrounding a protein and calculate the density inside that shell (i.e. the volume of the shell and the mass of protein and solvent inside it)
    
    
    #if pdb is a string then assume it is a string that refers to a pdb file in
    #the cwd. Create a MDA Universe of this pdb. Select protein atoms.
    if type(pdb) is str:
        U = mda.Universe(pdb)
        protein = U.select_atoms("protein")
        
    #if pdb file is not a string, assume it is already an MDA Universe.
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

    inner = U.select_atoms(f"(resname SOL and around {zone} protein) or protein")
    outer = U.select_atoms(f"resname SOL and not around {zone} protein")
    mass = np.sum(inner.masses)
    
    inner_positions = inner.positions
    outer_positions = outer.positions
    minpos = np.min(outer_positions, axis = 0) - leeway
    maxpos = np.max(outer_positions, axis = 0) + leeway
    myxrange = np.arange(minpos[0], maxpos[0], step)
    myyrange = np.arange(minpos[1], maxpos[1], step)
    myzrange = np.arange(minpos[2], maxpos[2], step)
    
    cnt = 0
    #If included list, excluded list, Delaunay, and Convex Hull all False
    for m in myzrange:
        #itertools.products finds the set of all ordered products between x, y, and z ranges. 
        #I.e. the below gives a segment of the full 3D grid for a specific z value.
        #All points in this 'slice' are loaded as temporary points
        pts_tmp = np.array(list(itertools.product(myxrange, myyrange, np.array([m]))))
        
        #For each point in each slice the minimum distance to the nearest protein and water atoms are found
        dist_i = np.min(S.cdist(pts_tmp, inner_positions), axis=1)
        dist_o = np.min(S.cdist(pts_tmp, outer_positions), axis=1)
        
        #include all the voxels of the slice for which distance to protein is 
        #less than distance to water in the protein volume
        cnt += np.sum(dist_i<=dist_o)
        
    inner_vol = cnt*(step**3)
    density = (mass/inner_vol)*1.6605402
    
    return mass*1.6605402, inner_vol, density


def shell_densities(u, zones = np.linspace(1.5,4,41), step = 0.5):

    masses = []
    inner_vols = []
    
    # for each shell, calculate the volume and mass associated with protein and volume within the shell
    for zone in zones:
        mass, inner_vol, density = protein_zone_density(u, step=step, leeway=5, zone = zone, rotate = False, internal_waters = False, internal_water_method = "positions")
        masses.append(mass)
        inner_vols.append(inner_vol)
    
    # by subtracting the inner mass and volume from the outer mass and volume and dividing, the solvent density within the shell can be determined
    shell_density = [
        (masses[i+1] - masses[i]) / (inner_vols[i+1] - inner_vols[i])
        if (inner_vols[i+1] - inner_vols[i]) != 0 else np.nan
        for i in range(len(inner_vols) - 1)
    ]
    return shell_density

def calc_rdfs_water(file, dist_range = (1.5,4), n_bins = 41, norm="none", step = 0.5):    
    
    zones = np.linspace(dist_range[0], dist_range[1], n_bins+1)
    u = mda.Universe(file)
    
    water = u.select_atoms("resname SOL or resname WAT")
    rdf_ = rdf.InterRDF(water,water,
                        nbins=n_bins, 
                        range=dist_range,  # distance
                        norm=norm)

    irdf = rdf_.run()
    edges = rdf_.results.edges

    inner_vols = []
    
    # for each shell, calculate the volume and mass associated with protein and volume within the shell
    

    for zone in zones:
        _, inner_vol, _ = protein_zone_density(u, step=step, leeway=5, zone = zone, rotate = False, internal_waters = False, internal_water_method = "positions")
        inner_vols.append(inner_vol)
    
    shell_volumes = [
        (inner_vols[i+1] - inner_vols[i])
        if (inner_vols[i+1] - inner_vols[i]) != 0 else np.nan
        for i in range(len(inner_vols) - 1)
    ]
    
    rdfs = rdf_.results.rdf
    rdfs_adj = rdfs/shell_volumes 
    
    return rdfs, rdfs_adj

def calc_rdfs_protein_water(file, dist_range = (1.5,4), n_bins = 75, norm="none", step=0.5):    
    u = mda.Universe(file)
    
    protein = u.select_atoms("protein")
    zones = np.linspace(dist_range[0], dist_range[1], n_bins+1)
    u = mda.Universe(file)
    
    water = u.select_atoms("resname SOL or resname WAT")
    rdf_ = rdf.InterRDF(protein,water,
                        nbins=n_bins, 
                        range=dist_range,  # distance
                        norm=norm)

    irdf = rdf_.run()
    edges = rdf_.results.edges

    inner_vols = []
    
    # for each shell, calculate the volume and mass associated with protein and volume within the shell
    

    for zone in zones:
        _, inner_vol, _ = protein_zone_density(u, step=step, leeway=5, zone = zone, rotate = False, internal_waters = False, internal_water_method = "positions")
        inner_vols.append(inner_vol)
    
    shell_volumes = [
        (inner_vols[i+1] - inner_vols[i])
        if (inner_vols[i+1] - inner_vols[i]) != 0 else np.nan
        for i in range(len(inner_vols) - 1)
    ]
    
    rdfs = rdf_.results.rdf
    rdfs_adj = rdfs/shell_volumes 
    
    return rdfs, rdfs_adj
