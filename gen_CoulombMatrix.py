#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 Hao Ren <renh.cn@gmail.com>
#
# Distributed under terms of the LGPLv3 license.

"""
Generate Coulomb matrices for the snapshots with corresponding IR_raw and 
xyz_standard both exists.

The Coulomb matrix is defined as [M. Rupp et al. PRL, 108, 058301(2012)]:
             - 0.5 * Z_I ^ 2.4,         for I = J
    M_{IJ} = |
             - Z_I*Z_J / |R_I - R_J|,   for I != J

We will calculate M as C / D, where C and D are the charge and distance matrices,
respectively.
    1) charge matrix C (calculated once before loop over snapshots):
        a) diagonal elements:  0.5 * Z_I ^ 2.4
        b) off-diagonal elements: Z_I * Z_J
    2) distance matrix D (depends on snapshots):
        a) diagonal elements: 1, to be dividable
        b) off-diagonal elements: the distance between atoms I and J

"""
import os
import numpy as np
from scipy.spatial.distance import pdist

atomic_charge = {
        'H': 1,
        'C': 6,
        'N': 7,
        'O': 8,
        }

def read_xyz(fname):
    with open(fname, 'r') as fh:
        natoms = int(fh.readline())
        coord = np.zeros([natoms,3])
        fh.readline()
        for ia in range(natoms):
            coord[ia] = [float(x) for x in fh.readline().split()[1:]]
        return coord

def write_xyz(fname, atoms, coords):
    natoms = len(atoms)
    if len(coords) != natoms:
        print('Inconsistent atom labels and coordinates')
        return
    with open(fname, 'w') as fh:
        fh.write('{}\n\n'.format(natoms))
        for i in range(natoms):
            fh.write('{:<4s}{:12.5f}{:12.5f}{:12.5f}\n'.format(
                atoms[i], *(coords[i])
                ))
    return


# read atomic labels/orders from one of the xyz files
# and generate the charge matrix C
with open('xyz_standard/s000001.xyz', 'r') as fh:
    lines = fh.readlines()
    N = int(lines[0])
    # only NMA atoms would be read
    atoms = [l.split()[0] for l in lines[2:14]]

atoms.extend(['H', 'H', 'H', 'O']) #append 3 H and 1 O for C=O and N-H HB
charges = [float(atomic_charge.get(atom)) for atom in atoms]
C = np.outer(charges, charges)
np.fill_diagonal(C, 0.5 * np.power(charges, 2.4))

DM = []
# generate distance matrix D and calculate M
for isnap in range(110000):
    f_xyz = 'xyz_standard/s{:06d}.xyz'.format(isnap)
    f_IR = 'IR_raw/s{:06d}.dat'.format(isnap)
    if not(os.path.exists(f_xyz) and os.path.exists(f_IR)):
        print('##{}: data not complete, continue...'.format(isnap))
        continue
    coords = read_xyz(f_xyz)
    this_coords = coords[:12] # NMA coords
    
    # find the nearer H from the three nearest water molecules
    n_water_mols = (len(coords) - 12) // 3
    coords_water = np.zeros([n_water_mols, 3, 3])
    for i in range(n_water_mols):
        coords_water[i] = coords[12+i*3:15+i*3]

    coord_H4 = this_coords[7]
    coord_O1 = this_coords[5]


    # for O1
    dists = coords_water[:,1:,:] - coord_O1
    dists = np.sum(dists*dists, axis=-1)
    dists = np.sqrt(dists)
    dist = np.min(dists, axis=-1)
    sorted_idx_H = np.argsort(dist)
    for i in range(3):
        this_H_idx = sorted_idx_H[i]
        nearer_H_idx = np.argsort(dists[this_H_idx], axis=-1)[0]
        this_coords = np.vstack(
            (this_coords, coords_water[this_H_idx, nearer_H_idx+1])
                )

    #for H4
    dist = coords_water[:,0,:] - coord_H4
    dist = np.sum(dist*dist, axis=-1)
    dist = np.sqrt(dist)
    sorted_idx_O = np.argsort(dist)
    # append the nearest two O atoms from H4
    this_coords = np.vstack(
        (this_coords, coords_water[sorted_idx_O[0],0])
            )

    #write_xyz('test.xyz', atoms, this_coords)

    # since all the snapshots consists of the same atoms, we can only use
    # the distance matrix to represent the structures.
    D = pdist(this_coords)
    DM.append(D)
    if not (isnap + 1) % 10000:
        print('{} snapshots processed.'.format(isnap+1))

np.save('DM.npy', DM)


