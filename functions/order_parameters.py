import numpy as np
import scipy.spatial.distance as S

def calc_otos(waters):
    """
    Calculates the Orientational tetrahedral order parameter for a collection of water oxygen positions.

    Reference: 
        
    Parameters
    ----------
    waters : NumPy Array
        2D NumPy array of water atom oxygen positions (n,3).

    Returns
    -------
    otos : NumPy array
        NumPy array of orientational tetrahedral order parameters for each water oxygen in input array.

    """
    if len(waters) <= 4:
        otos = np.empty(waters.shape[0])
        otos.fill(np.nan)
        return otos
    
    def cos_angle_between_vectors(v1, v2):
        # Normalize the vectors
        v1_norm = np.linalg.norm(v1, axis=1)
        v2_norm = np.linalg.norm(v2, axis=1)

        v1 = v1 / v1_norm[:, np.newaxis] #RuntimeWarning: invalid value in divide
        v2 = v2 / v2_norm[:, np.newaxis] #RuntimeWarning: invalid value in divide
        
        # Directly compute the cosine of the angle as the dot product of normalized vectors
        cos_angle = np.einsum('ij,ij->i', v1, v2)
        
        # Clamp the cosine values to avoid numerical issues
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return cos_angle
    
    dists = S.cdist(waters, waters)
    np.fill_diagonal(dists, np.inf)
    
    # Pre-sort distances to get the indices of nearest neighbors
    sorted_indices = np.argsort(dists, axis=1)
    
    num_neighbours = 4
    neighbourhood = sorted_indices[:, :num_neighbours]
    
    # Preallocate otos array
    otos = np.zeros(waters.shape[0])
    
    # Precompute constants
    third = 1 / 3
    
    for i, neighbours in enumerate(neighbourhood):
        
        # Get central water for which we find angles around
        central_water = waters[i]
        # Get pairs of vectors for cosine angle calculation
        vec_pairs = [(waters[neighbours[j]], waters[neighbours[j+1]]) for j in range(0, num_neighbours-1)]
        # Get vectors relative to central water
        centred_vec_pairs = vec_pairs - central_water
        
        # Calculate cosine of angles between vectors
        cos_angles = cos_angle_between_vectors(np.array([pair[0] for pair in centred_vec_pairs]),
                                               np.array([pair[1] for pair in centred_vec_pairs]))
        
        # Apply the formula using cosine values directly
        otos[i] = 1 - 3/8*np.sum([(cos_angles[j]+1/3)**2 for j in range(3)])
      
    return otos

def calc_ttos(waters):
    """
    Calculates the Translational tetrahedral order parameter for a collection of water oxygen positions.

    Reference: 
        
    Parameters
    ----------
    waters : NumPy Array
        2D NumPy array of water atom oxygen positions (n,3).

    Returns
    -------
    ttos : NumPy array
        NumPy array of translational tetrahedral order parameters for each water oxygen in input array.

    """
    if len(waters) <= 4:
        ttos = np.empty(waters.shape[0])
        ttos.fill(np.nan)
        return ttos
    
    dists = S.cdist(waters, waters)
    np.fill_diagonal(dists, np.inf)
    
    # Pre-sort distances to get the indices of nearest neighbors
    sorted_indices = np.argsort(dists, axis=1)
    
    num_neighbours = 4
    neighbourhood = sorted_indices[:, :num_neighbours+1]
    
    # Preallocate otos array
    ttos = np.zeros(waters.shape[0])
    
    # Precompute constants
    third = 1 / 3
        
    for i, neighbours in enumerate(neighbourhood):
        
        # Get central water for which we find angles around
        central_water = waters[neighbours[0]]
        
        # Get mean of radial distances
        r = np.linalg.norm(waters[neighbours][1:]-central_water, axis=1)
        r_mean = np.mean(r)
    
        # Calculate cosine of angles between vectors
        
        tto = 1 - third*np.sum((r-r_mean)** 2/(4*np.pi*r**2))
        ttos[i] = tto
      
    return ttos

def calc_lsis(waters):
    
    """
    Calculates the Local Structure Index (LSI) for every water oxygen in a collection of water oxygen positions.
    LSI quantifies the distance between a first and second hydration shell surrounding a water molecule.

    Reference: 
        
    Parameters
    ----------
    waters : NumPy Array
        2D NumPy array of water atom oxygen positions (n,3).

    Returns
    -------
    lsis : NumPy array
        NumPy array of LSIs for each water oxygen in input array.

    """
    if len(waters) <= 4:
        lsis = np.empty(waters.shape[0])
        lsis.fill(np.nan)
        return lsis
    
    dists = S.cdist(waters, waters)
    np.fill_diagonal(dists, np.inf)
    
    # Pre-sort distances
    sorted_indices = np.argsort(dists, axis=1)
    
        
    sorted_dists = np.array([dists[i][sorted_indices[i]] for i in range(len(dists))])
    
    shell_sums = [[sorted_dist[i+1] - sorted_dist[i] for i in range(len(dists)) if sorted_dist[i] < 3.7] for sorted_dist in sorted_dists]
    
    lsis = [sum([(deltas[i] - np.mean(deltas))**2 for i in range(len(deltas))])/len(deltas) if len(deltas)>0 else 0 for deltas in shell_sums]
    
    return lsis        
            