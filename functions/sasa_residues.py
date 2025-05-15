import numpy as np
import biobox as bb

def sasa(pdb, targets=[], probe=1.4, n_sphere_point=960, threshold=0.05, return_count = False):
#from Matteo        
#compute the accessible surface area using the Shrake-Rupley algorithm ("rolling ball method")

#:param M: any biobox object
#:param targets: indices to be used for surface estimation. By default, all indices are kept into account.
#:param probe: radius of the "rolling ball"
#:param n_sphere_point: number of mesh points per atominch
#:param threshold: fraction of points in sphere, above which structure points are considered as exposed
#:returns: accessible surface area in A^2
#:returns: mesh numpy array containing the found points forming the accessible surface mesh
#:returns: IDs of surface points

    if pdb.endswith(".gro"):
        gro = pdb
        M = bb.Molecule()
        M.import_gro(gro)
    else:
        M = bb.Molecule()
        M.import_pdb(pdb)
    import biobox.measures.interaction as I

    #make sure that everything is collected as a Structure object, and radii are available
    this_inst = type(M).__name__
    if this_inst == "Multimer":
        M = M.make_molecule()

    elif this_inst in ["Assembly", "Polyhedra"]:
        M = M.make_structure()
        
    pos_p, idx_p = M.query('resname != ["SOL", "NA", "CL"]', get_index = True)
    M = M.get_subset(idx_p)

    if len(targets) == 0:
        targets = range(0, len(M.points), 1)

    # getting radii associated to every atom
    radii = M.data['radius'].values

    if threshold < 0.0 or threshold > 1.0:
        raise Exception("ERROR: threshold should be a floating point between 0 and 1!")

    # create unit sphere points cloud (using golden spiral)
    pts = []
    inc = np.pi * (3 - np.sqrt(5))
    offset = 2 / float(n_sphere_point)
    for k in range(int(n_sphere_point)):
        y = k * offset - 1 + (offset / 2)
        r = np.sqrt(1 - y * y)
        phi = k * inc
        pts.append([np.cos(phi) * r, y, np.sin(phi) * r])

    sphere_points = np.array(pts)
    const = 4.0 * np.pi / len(sphere_points)

    contact_map = I.distance_matrix(M.points, M.points)

    asa = 0.0
    surface_atoms = []
    mesh_pts = []
    indices = []
    cnts = []
    #cnts = {}
    # compute accessible surface for every atom
    for i in targets:

        # place mesh points around atom of choice
        mesh = sphere_points * (radii[i] + probe) + M.points[i]

        # compute distance matrix between mesh points and neighboring atoms
        test = np.where(contact_map[i, :] < radii.max() + probe * 2)[0]
        # don't consider atom mesh points surround as neighbour
        test = np.delete(test, np.where(test == i)) ### ADDED LINE ###
        neigh = M.points[test]
        adj = radii[test][:, np.newaxis]
        dist = I.distance_matrix(neigh, mesh) - adj

        # lines=atoms, columns=mesh points. Count columns containing values greater than probe*2
        # i.e. allowing sufficient space for a probe to fit completely
        cnt = 0
        for m in range(dist.shape[1]):
            if not np.any(dist[:, m] < probe):
                cnt += 1
                mesh_pts.append(mesh[m])

        indices.extend(cnt*[i])
        
        # calculate asa for current atom, if a sufficient amount of mesh
        # points is exposed (NOTE: to verify)
        if cnt >= n_sphere_point * threshold:
            surface_atoms.append(i)
            asa += const * cnt * (radii[i] + probe)**2
            cnts.append(cnt)
            #cnts[i] = cnt

    if return_count == True:
        return asa, np.array(mesh_pts), np.array(surface_atoms), np.array(indices), np.array(cnts)
    else:
        return asa, np.array(mesh_pts), np.array(surface_atoms), np.array(indices)