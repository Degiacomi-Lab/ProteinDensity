# Protein Density

A method to calculate the density of proteins in explicit water.
See publication in the paper folder.
For associated trained regressor, see https://github.com/Degiacomi-Lab/DensiTree

## Overview of functions

### protein_volume_boxes.py
A 3-D grid is constructed around the extremes of the coordinates of the protein structure in each Cartesian axis direction, which is then split into a number of smaller 'boxes' of grids. The volume is simply calculated by calculating by treating each grid point as the centre of a voxel and defining whether a voxel is part of the protein or solvent based on which atom type it is closest to. The summed voxel volumes gives the overall protein volume, from which protein density is found.

### protein_excess_volume_boxes.py
Calculates the 'excess' protein volume that surrounds the protein surface but is outside of the Van der Waals radii of both the nearest protein and water atoms. Interestingly, when this excess volume is removed, a relationship between protein mass and density --- sometimes observed in literature --- (re)emerges.

### protein_volume_residues.py
Calculates the volume of each residue via a similar principle to protein_volume_boxes: i.e., for each residue a surrounding grid is constructed, and the volume of the residue is defined as the summed volume of all voxels that are nearer to residue protein atoms than any other atoms (including solvent and protein atoms belonging to other residues). Summing the residue volumes should give an overall volume very close or identical to the protein volume as calculated by protein_volume_boxes.

### protein_volume_residues_surf_int.py
Built on the protein_volume_residues function. Calculates the volume of each residue via the method outlined above, while also giving an indication as to the level of solvent exposure of a residue, as defined by the Shrake-Rupley Rolling Ball algorithm implementation in sasa_residues.py.

### sasa_residues.py
Similar to the sasa function found in the [biobox python package](https://github.com/Degiacomi-Lab/biobox), but adapted to give more detailed data on the solvent exposure of each protein atom in order to be useful to the protein_volume_residues_surf_int.py function.

### find_internal_waters.py
Clusters water molecules surrounding a protein using DBSCAN, with the clusters filtered to reveal which represent water molecules in the protein interior.

### order_parameters.py
Methods for calculating three order parameters: the local structure index (LSI), orientational tetrahedral order parameter (_q_), and the translational tetrahedral order parameter (S<sub>k</sub>).



