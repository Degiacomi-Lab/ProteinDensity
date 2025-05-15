import numpy as np
from sklearn.cluster import DBSCAN as DBSCAN
import MDAnalysis as mda
import scipy.spatial.distance as S


def find_internal_waters(u, tolerance=3):
    #u is MDAnalysis universe or pdb file
    #returns a list of the resnums of internal water molecules for currently selected
    #timestep of MDA universe
    #tolerance defines the maximum distance a water molecule can be from a protein atom while
    #still being considered an 'internal' water molecule. 'Coupled' water molecule may be a better term.
    
    if type(u) != mda.core.universe.Universe:
        try:
            u = mda.Universe(u)
        except:
            # assume u is a MDAnalysis atom group
            u = u
        
    #Select water oxygens

    waters = u.select_atoms("name OW and around 6 protein")
    if len(waters) == 0:
        waters = u.select_atoms("(name O and resname WAT) and around 6 protein")
    #find positions of water oxygens
    water_positions = waters.positions
    #cluster water molecules, so that bulk water makes main cluster
    #water molecules far from bulk will be embedded inside protein
    
    protein_positions = u.select_atoms("protein").positions
    
    db = DBSCAN(eps=4, min_samples=1).fit(water_positions) #interesting at eps=3.2, min_samples=2
    labels = db.labels_
    cluster_one = np.where(labels==1)[0]
    
    #return empty lists if no internal waters found
    if len(cluster_one) == 0 :
        return [], []
    
    internal_resnums = []
    internal_positions = []
    for i in range(1, np.max(labels)+1):
        cluster_resnums = waters.resnums[labels==i]
        cluster_positions = waters.positions[labels==i]
        internal_resnums.append(cluster_resnums)
        internal_positions.append(cluster_positions)
    
    internal_positions = np.concatenate(internal_positions, axis = 0)
    internal_resnums = np.concatenate(internal_resnums, axis = 0)
    
    #filter to only include water molecules within tolerance angstoms of nearest protein atom
    distances = S.cdist(internal_positions, protein_positions)
    internal_positions = internal_positions[np.min(distances, axis=1)<tolerance]
        
    return internal_resnums, internal_positions

