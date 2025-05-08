# -*- coding: utf-8 -*-


# Imports
import scipy.spatial.distance as dist
import numpy as np
import mdtraj as md
import MDAnalysis as mda
import sys
import scipy.spatial.distance as sdist
import pandas as pan
import pickle
import os


#%%
class Protein():
    def __init__(self, protein, keyword="sequence"):
        keyword = self.keyword
        if keyword == "sequence":
            self.sequence = protein
        elif keyword == "structure" or keyword == "both":
            self.pdb = protein

        self.ATOMIC_RADII_ELEMENTS = {'H'   : 0.120, 'He'  : 0.140, 'Li'  : 0.076, 'Be' : 0.059,
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
    
        #From ProteinVolume1.3
        #ATOMIC_RADII_NAMES = {}
        #with open('radii.rad', 'r') as file:
        #    for line in file:
        #        atom_name, radius = line.split()
        #        ATOMIC_RADII_NAMES[atom_name] = float(radius)
    
        #adjusted to change HW1, HW2
        self.ATOMIC_RADII_NAMES = {'N': 1.824, 'H': 0.6, 'CA': 1.908, 'HA': 1.1, 'CB': 1.908, 
                              'HB1': 1.487, 'HB2': 1.487, 'HB3': 1.487, 'C': 1.908, 'O': 1.6612,
                              'CG': 1.908, 'HG1': 0.0, 'HG2': 1.487, 'CD': 1.908, 'HD1': 1.459, 
                              'HD2': 1.459, 'NE': 1.824, 'HE': 0.6, 'CZ': 1.908, 'NH1': 1.824,
                              'HH11': 0.6, 'HH12': 0.6, 'NH2': 1.824, 'HH21': 0.6, 'HH22': 0.6,
                              'OD1': 1.6612, 'OD2': 1.6612, 'ND2': 1.824, 'HD21': 1.487, 
                              'HD22': 1.487, 'SG': 2.0, 'HG': 0.0, 'OE1': 1.6612, 'OE2': 1.6612,
                              'HE2': 1.459, 'NE2': 1.824, 'HE21': 0.6, 'HE22': 0.6, 'HA1': 1.1, 
                              'HA2': 1.1, 'ND1': 1.824, 'CE1': 1.908, 'HE1': 1.459, 'CD2': 1.908,
                              'HB': 1.487, 'CG2': 1.908, 'HG21': 1.487, 'HG22': 1.487, 'HG23': 1.487,
                              'CG1': 1.908, 'HG11': 1.487, 'HG12': 1.487, 'CD1': 1.908, 'HD3': 1.487,
                              'HD11': 1.487, 'HD12': 1.487, 'HD13': 1.487, 'HD23': 1.487, 'CE': 1.908, 
                              'NZ': 1.824, 'HZ1': 0.6, 'HZ2': 1.459, 'HZ3': 1.459, 'SD': 2.0, 'HE3': 1.459, 
                              'HZ': 1.459, 'CE2': 1.908, 'OG': 1.721, 'OG1': 1.721, 'NE1': 1.824, 
                              'CZ2': 1.908, 'CH2': 1.908, 'HH2': 1.459, 'CZ3': 1.908, 'CE3': 1.908,
                              'OH': 1.721, 'HH': 0.0, 'HG13': 1.487, 'OC1': 1.6612, 'OC2': 1.6612, 
                              'OXT': 1.6612, 'HSG': 0.6, 'HN1': 0.6, 'HN2': 0.6, 'CH3': 1.908, 
                              'HH31': 1.487, 'HH32': 1.487, 'HH33': 1.487, 'H1': 0.6, 'H2': 1.359,
                              'H3': 0.6, 'P': 2.1, 'O1P': 1.6612, 'O2P': 1.6612, "O5'": 1.721, 
                              "C5'": 1.908, "H5'1": 1.387, "H5'2": 1.387, "C4'": 1.908, "H4'": 1.387,
                              "O4'": 1.6837, "C1'": 1.908, "H1'": 1.287, 'N9': 1.824, 'C8': 1.908,
                              'H8': 1.359, 'N7': 1.824, 'C5': 1.908, 'C6': 1.908, 'N6': 1.824, 
                              'H61': 0.6, 'H62': 0.6, 'N1': 1.824, 'C2': 1.908, 'N3': 1.824, 
                              'C4': 1.908, "C3'": 1.908, "H3'": 1.387, "C2'": 1.908, "H2'1": 1.387,
                              "H2'2": 1.487, "O3'": 1.721, 'H3T': 0.0, 'H5T': 0.0, 'H6': 1.409, 
                              'H5': 1.459, 'N4': 1.824, 'H41': 0.6, 'H42': 0.6, 'O2': 1.6612, 
                              'O6': 1.6612, 'N2': 1.824, 'H21': 0.6, 'H22': 0.6, 'C7': 1.908, 
                              'H71': 1.487, 'H72': 1.487, 'H73': 1.487, 'O4': 1.6612, "O2'": 1.721, 
                              "HO'2": 0.0, 'HW': 0.0, 'OW': 1.6612, 'HW1': 1.2, 'HW2': 1.2}
        
    def extract_residues(self, fmt = "", chain = "X"):
    
        #Argument:
        #seq, str: either a list of one-letter residue codes, three-letter residue codes, or a PDB file, or a FASTA file.
        #Return:
        #residues, str: a list of three-letter residue codes
        
        seq = self.sequence
        assert type(seq) is str, f"seq entry: \"{seq}\" is not valid.\nseq should be a string, either pointing to a PDB file, a fasta file, or a string of one-letter or three-letter amino acid residue codes."
        
        # define dictionary of one and three letter amino acid residue codes, required if seq is not a PDB file
        all_residues_dict = {
        'V': 'VAL', 'R': 'ARG', 'S': 'SER', 'L': 'LEU', 
        'N': 'ASN', 'C': 'CYS', 'T': 'THR', 'D': 'ASP',
        'Q': 'GLN', 'K': 'LYS', 'M': 'MET', 'G': 'GLY', 
        'P': 'PRO', 'Y': 'TYR', 'E': 'GLU', 'A': 'ALA', 
        'H': 'HIS', 'F': 'PHE', 'I': 'ILE', 'W': 'TRP'}
        
        # if seq is a PDB file
        if seq.endswith(".pdb") or fmt == "pdb":
            # initiate MDAnalysis Universe of chain of seq, to easily extract three-letter residue codes
            if not seq.endswith(".pdb"): seq+=".pdb"
            u = mda.Universe(seq).select_atoms(f"protein and chainid {chain}")
            residues = u.residues.resnames
            residues = np.where((residues=="HIE") | (residues=="HID") | 
                                        (residues=="HIP"), "HIS", residues)
            return residues
        
        # elif seq is a fasta file
        elif seq.endswith(".fasta") or fmt == "fasta":
            # get resnames in single letter code
            residues_fasta = open(seq, "r").read().split("\n")[1]
            # convert single letter code to three letter codes
            residues = [all_residues_dict[res] for res in residues_fasta]
            residues = np.where((residues=="HIE") | (residues=="HID") | 
                                        (residues=="HIP"), "HIS", residues)
            return residues
        
        # elif seq is a list of three-letter amino acid codes
        elif fmt == "threeletter" or (((sum(list(map(lambda residue:residue in all_residues_dict.values(), [seq[i:i+3] for i in range(0,len(seq)-2, 3)])))) == len(seq)//3) and len(seq)%3 == 0):
            residues =  [seq[i:i+3] for i in range(0,len(seq)-2, 3)]
            residues = np.where((residues=="HIE") | (residues=="HID") | 
                                        (residues=="HIP"), "HIS", residues)
            return residues
            
        # else assume seq is a list of one-letter amino acid codes
        elif fmt == "oneletter" or (sum(list(map(lambda residue:residue in all_residues_dict.keys(), seq))) == len(seq)):
            residues = [all_residues_dict[res] for res in seq]
            return residues
        
        else:
            print("Incomprehensible seq string: {seq}")
            return
        
    def est_ssbonds(self):
        tol = 0.05
        u = mda.Universe(self.pdb)
        protein = u.select_atoms("protein")
        residues =  protein.atoms
        atoms =  protein.atoms.names
        positions = protein.atoms.positions
        sgs = positions[atoms == "SG"]
        indices = protein.atoms.indices[atoms == "SG"]
        dist_matrix = np.tril(dist.cdist(sgs, sgs))
        dist_matrix = dist_matrix[np.logical_and(dist_matrix > 2.05-tol, dist_matrix<2.05+tol)]
        return {"Disulfide bonds": len(dist_matrix)}
    

    def aspect_ratio(self):
        u = mda.Universe(self.pdb)
        protein = u.select_atoms("protein")
        #e1,e2,e3 = protein.principal_axes()
        protein.align_principal_axis(0, [0,0,1])
        protein.align_principal_axis(1, [0,1,0])
        protein_aligned = protein
        pos = protein_aligned.positions
        
        #get extent along each principal component
        mins = np.min(pos, axis = 0)
        maxs = np.max(pos, axis = 0)
        diffs = maxs-mins
    
        #calculate the aspect ratio
        aratio = max(diffs)/min(diffs)

        return {"Aspect ratio": aratio}
    
    def amino_acid_composition(self, surface_atoms):
        #standard amino acid residues with HIS variations
        residues = ['VAL','ARG','SER','LEU','ASN','CYS','THR','ASP','GLN',
                    'LYS','MET','GLY','PRO','TYR','GLU','ALA','HIS', 
                    'PHE','ILE','TRP']
        
        #definition of protein categories
        charged = ["LYS", "ARG", "ASP", "GLU", "HIS"]
        hydrophobic = ["ALA", "VAL", "LEU", "ILE", "PHE","MET", "TRP"]
        other = [residue for residue in residues if residue not in hydrophobic + charged]
        acidic_charged = ["ASP", "GLU"]
        basic_charged = ["ARG", "HIS", "LYS"]
        aliph_hyd = ["ALA", "VAL", "LEU", "ILE",  "MET"]
        arom_hyd = ["PHE", "TRP"]
        arom = ["PHE", "TRP", "TYR", "HIS"]
        aliph = ['VAL','ARG','SER','LEU','ASN','CYS','THR','ASP','GLN',
                    'LYS','MET','GLY','PRO','GLU','ALA','ILE']
        
        cats = {"Hydrophobic %":hydrophobic, "Charged %":charged, "Other %":other, 
                "Acidic charged %":acidic_charged, "Basic charged %":basic_charged, 
                "Aliph. hydrophobic %":aliph_hyd, "Arom. hydrophobic %":arom_hyd, 
                "Aromatic %":arom, "Aliphatic %":aliph}
        
        #get selection syntax for surface and interior of protein
        #selection_surface = "protein and" + " or".join(f" index {index}" for index in surface_atoms)
        #selection_interior = "protein and not (" + " or".join(f" index {index}" for index in surface_atoms) + ")"
        
        #get protein structure file
        pdb = self.pdb

        #calculate amino acid residue composition of protein (percentages of each amino acid)
        protein_atoms = mda.Universe(pdb).select_atoms("protein").indices
        interior_atoms = list(set(protein_atoms) - set(surface_atoms))
                              
        protein_residues = mda.Universe(pdb).select_atoms("protein").residues.resnames
        #correct for HIS variants
        protein_residues = np.where((protein_residues=="HIE") | (protein_residues=="HID") | 
                                    (protein_residues=="HIP"), "HIS", protein_residues)
        amino_acid_composition = {res+" %":100*sum(protein_residues==res)/len(protein_residues) for res in residues}
        
        #calculate residue composition of entire protein
        protein_resnames = mda.Universe(pdb).select_atoms("protein").resnames
        protein_resnames = np.where((protein_resnames=="HIE") | (protein_resnames=="HID") | 
                                    (protein_resnames=="HIP"), "HIS", protein_resnames)
        protein_values = list(map(lambda cat: 100*sum([1 if res in cat else 0 for res in protein_resnames])/len(protein_resnames), cats.values()))
        protein_composition = {cat_name:value for cat_name, value in zip(cats.keys(), protein_values)}
        amino_acid_composition.update(protein_composition)
        
        #calculate residue composition of protein surface
        surface_resnames = mda.Universe(pdb).atoms[surface_atoms].resnames
        #surface_resnames = mda.Universe(pdb).select_atoms(selection_surface).resnames
        surface_resnames = np.where((surface_resnames=="HIE") | (surface_resnames=="HID") | 
                                    (surface_resnames=="HIP"), "HIS", surface_resnames)
        surface_values = list(map(lambda cat: 100*sum([1 if res in cat else 0 for res in surface_resnames])/len(surface_resnames), cats.values()))
        surface_composition = {cat_name+" surface":value for cat_name, value in zip(cats.keys(), surface_values)}
        amino_acid_composition.update(surface_composition)

        #calculate residue composition of protein interior
        interior_resnames = mda.Universe(pdb).atoms[interior_atoms].resnames
        #interior_resnames = mda.Universe(pdb).select_atoms(selection_interior).resnames
        interior_resnames = np.where((interior_resnames=="HIE") | (interior_resnames=="HID") | 
                                    (interior_resnames=="HIP"), "HIS", interior_resnames)
        interior_values = list(map(lambda cat: 100*sum([1 if res in cat else 0 for res in interior_resnames])/len(interior_resnames), cats.values()))
        interior_composition = {cat_name+" interior":value for cat_name, value in zip(cats.keys(), interior_values)}
        amino_acid_composition.update(interior_composition)

        
        return amino_acid_composition
    
    def est_charge_mass(self):
        # get total net charge estimate
        
        # usual protein charges at physiological pH
        neg = ["GLU", "ASP"]
        pos = ["ARG", "LYS", "HIP"]
        
        # get residue composition
        pdb = self.pdb
        protein_residues = mda.Universe(pdb).select_atoms("protein").residues.resnames

        # estimate total charge as sum of expected charges
        charge = sum([-1 if residue in neg else 1 if residue in pos else 0 for residue in protein_residues])
        
        # estimate total mass
        mass = sum(mda.Universe(pdb).select_atoms("protein").masses)
        
        return {"Net charge": charge, "Mass": mass}
    
    def est_secondary_structure(self):
        pdb = self.pdb
        traj = md.load(pdb)
        dssp = md.compute_dssp(traj)[0]
        
        coil_percent = 100*sum(dssp=="C")/len(dssp)
        strand_percent = 100*sum(dssp=="E")/len(dssp)
        helix_percent = 100*sum(dssp=="H")/len(dssp)
        
        return {"Coil percent": coil_percent, "Strand percent": strand_percent, "Helix percent":helix_percent}

    def calc_sasa(self, targets=[], probe=1.4, n_sphere_point=960, threshold=0.05, radius = "element"):
    #from Matteo        
    #compute the accessible surface area using the Shrake-Rupley algorithm ("rolling ball method")

    #:param M: any biobox object
    #:param targets: indices to be used for surface estimation. By default, all indices are kept into account.
    #:param probe: radius of the "rolling ball"
    #:param n_sphere_point: number of mesh points per atominch
    #:param threshold: fraction of points in sphere, above which structure points are considered as exposed
    #:param radius: the type of VdW radius used, either 'element' or 'atomtype'
    #:returns: accessible surface area in A^2
    #:returns: mesh numpy array containing the found points forming the accessible surface mesh
    #:returns: IDs of surface points
    
        pdb = self.pdb
        u = mda.Universe(pdb)

        protein = u.select_atoms("protein")
        pos_p, idx_p = protein.positions, protein.indices

        if len(targets) == 0:
            targets = range(0, len(protein), 1)

        # getting radii associated to every atom
        
        #get vdw radii for protein and water
        prot_atom_names = protein.atoms.names
        prot_elements = protein.atoms.elements
        
        radii = []
        ATOMIC_RADII_ELEMENTS = self.ATOMIC_RADII_ELEMENTS
        ATOMIC_RADII_NAMES = self.ATOMIC_RADII_NAMES
        
        for atom_name, element in zip(prot_atom_names, prot_elements):
            
            if radius == "element":
                atom_radius = ATOMIC_RADII_ELEMENTS[element]*10
                radii.append(atom_radius)
        
            elif radius == "atomtype":
                try:
                    atom_radius = ATOMIC_RADII_NAMES[atom_name]
                except:
                    print(f"Warning: No radius entry for atom name {atom_name}.\nUsing element radius for {element} instead.")
                    atom_radius = ATOMIC_RADII_ELEMENTS[element]*10
                    
                radii.append(atom_radius)
        
        # convert radii list to array
        radii = np.array(radii)
        
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

        contact_map = sdist.cdist(protein.positions, protein.positions)

        asa = 0.0
        surface_atoms = []
        mesh_pts = []
        indices = []
        cnts = []
        #cnts = {}
        # compute accessible surface for every atom
        for i in targets:

            # place mesh points around atom of choice
            mesh = sphere_points * (radii[i] + probe) + protein.positions[i]

            # compute distance matrix between mesh points and neighboring atoms
            test = np.where(contact_map[i, :] < max(radii) + probe * 2)[0]
            # don't consider atom mesh points surround as neighbour
            test = np.delete(test, np.where(test == i)) ### ADDED LINE ###
            neigh = protein.positions[test]
            
            #return test, neigh, mesh, radii
            
            adj = radii[test][:, np.newaxis]
            dist = sdist.cdist(neigh, mesh) - adj

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
            if cnt > n_sphere_point * threshold:
                surface_atoms.append(i)
                asa += const * cnt * (radii[i] + probe)**2
                cnts.append(cnt)
                #cnts[i] = cnt

        return {"SASA": asa}, mesh_pts, surface_atoms, indices

    def radius_of_gyration(self):
        pdb = self.pdb
        Rg = mda.Universe(pdb).select_atoms("protein").radius_of_gyration()
        return {"Rg": Rg}

    def featurize(self):
        
        if self.keyword == "structure" or self.keyword == "both":
            protein = self.protein   
            # create dictionary to store structure_feats 
            structure_feats = {}
            
            # Add feature structure_feats to dictionary
            asa, _, surface_atoms, _ = protein.calc_sasa()
            structure_feats.update(asa)
            structure_feats.update(protein.est_ssbonds())
            structure_feats.update(protein.aspect_ratio())
            composition = protein.amino_acid_composition(surface_atoms)
            structure_feats.update(composition)
            structure_feats.update(protein.est_charge_mass())
            structure_feats.update(protein.est_secondary_structure())
            structure_feats.update(protein.radius_of_gyration())
            
            structure_feats = pan.DataFrame(structure_feats, index=[0])
            
            if self.keyword =="structure":
                return structure_feats
        
        elif self.keyword == "sequence" or self.keyword == "both":
            
            residues = self.extract_residues() # format ("fmt") and chain options
            # 20 standard amino acid residues
            all_residues = ['VAL','ARG','SER','LEU','ASN','CYS','THR','ASP','GLN',
                        'LYS','MET','GLY','PRO','TYR','GLU','ALA','HIS','PHE','ILE','TRP']
            
            col_names = [res + " %" for res in all_residues]
            
            residues = np.array(residues)
            
            col_values = [100*(len(residues[residues==res]))/len(residues) for res in all_residues]
                
            # create dictionary of values
            pdb_data = {col_name: col_value for (col_name, col_value) in zip(col_names, col_values)}
            
            # Add column values and column names from dictionary to a DataFrame
            sequence_feats = pan.DataFrame(data = pdb_data, index = [0])
            
            if self.keyword == "sequence":
                return sequence_feats
            
            elif self.keyword == "both":
                return {"sequence_feats": sequence_feats, "structure_feats": structure_feats}
    
    def predict(self, RF="default"):
        
        if self.keyword == "sequence":
            
            if RF == "default":
                with open("random_forests/RF_300_seq.pickle", "rb") as r_file:
                    RF_seq = pickle.load(r_file)
                
            #else:
                #assert
                
            feats = self.featurize()
            
            prediction = RF.predict(feats)[0]
            
            return prediction, feats
        
        elif self.keyword == "structure":
            
            feats = self.featurize()
            
            prediction = RF.predict(feats)[0]
            
            return prediction, feats
            
    

#%%

Protein.predict("ADTRYPGFC")

"""
#%%

if __name__ == "__main__":
    
    # load Random Forest Regressors
    
    with open("RF_300.pickle", "rb") as r_file:
        RF_structure = pickle.load(r_file)
        
    with open("RF_300_means.pickle", "rb") as r_file:
        RF_structure_means = pickle.load(r_file)
    
    with open("RF_300_seq.pickle", "rb") as r_file:
        RF_seq = pickle.load(r_file)
    
    #pdb = f"C:/Users/hkwf34/OneDrive - Durham University/Desktop/proteins/titin_pulling_300k/new_data_300k_nowater/2000_sim0_nowater.pdb"
    #feats, feats_seq = density_prediction(pdb)
    #feats.to_csv(f"TEST_structure_density_prediction_data.csv", index=False)
    #feats_seq.to_csv(f"TEST_sequence_density_prediction_data.csv", index=False)
    
    if len(sys.argv) == 2:
        pdb = sys.argv[-1]
        #pdb = f"C:/Users/hkwf34/OneDrive - Durham University/Desktop/proteins/titin_pulling_300k/new_data_300k_nowater/2000_sim0_nowater.pdb"
    
        feats, feats_seq = density_prediction(pdb)
        pdb_name = pdb.split(".")[0]
        feats.to_csv(f"{pdb_name}_structure_density_prediction_data.csv", index=False)
        feats_seq.to_csv(f"{pdb_name}_sequence_density_prediction_data.csv", index=False)

    elif len(sys.argv) > 2:
        
        files = sys.argv[1:]
        
        print(files)
        
        for i, pdb in enumerate(files):
            print(pdb)
            feats, feats_seq = density_prediction(pdb)            
            if i == 0:
                all_feats = feats
                all_feats_seq = feats_seq
            
            else:
                all_feats = pan.concat([all_feats, feats])
                all_feats_seq = pan.concat([all_feats_seq, feats_seq])
        
        directory = os.getcwd().split("/")[-1]
        all_feats.to_csv(f"{directory}_structure_density_prediction_data.csv", index=False)
        all_feats_seq.to_csv(f"{directory}_sequence_density_prediction_data.csv", index=False)
"""