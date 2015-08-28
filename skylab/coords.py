# -*-coding:utf8-*-

r"""
This file is part of SkyLab

Skylab is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


coords
======

Coordinate transformation methods and similar methods.

"""

import healpy as hp
import numpy as np

def rotate(ra1, dec1, ra2, dec2, ra3, dec3):
    r""" Rotate ra1 and dec1 in a way that ra2 and dec2 will exactly map
    onto ra3 and dec3, respectively. All angles are treated as radians.

    """

    # turn rightascension and declination into zenith and azimuth for healpy
    phi1 = ra1 - np.pi
    zen1 = np.pi/2. - dec1
    phi2 = ra2 - np.pi
    zen2 = np.pi/2. - dec2
    phi3 = ra3 - np.pi
    zen3 = np.pi/2. - dec3

    # rotate each ra1 and dec1 towards the pole
    x = np.array([hp.rotator.rotateDirection(
                    hp.rotator.get_rotation_matrix((dp, -dz, 0.))[0],
                    z, p) for z, p, dz, dp in zip(zen1, phi1, zen2, phi2)])

    # Rotate **all** these vectors towards ra3, dec3
    zen, phi = hp.rotator.rotateDirection(
                hp.rotator.get_rotation_matrix((-phi3, zen3, 0))[0],
                x[:,0], x[:,1])

    dec = np.pi/2. - zen
    ra = phi + np.pi

    return np.atleast_1d(ra), np.atleast_1d(dec)


