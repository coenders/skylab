# -*-coding:utf8-*-

r"""
Copyright (c) 2014, Stefan Coenders, coenders#icecube.wisc.edu
All rights reserved.

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


SkyLab

=====

SkyLab is a project for unbinned Likelihood analyses of point source searches
as performed by Neutrino Telescopes like ANTARES or IceCube.

Requirements
------------
    numpy (1.8 or higher needed for coordinate transformations)
    scipy (Versions earlier than 0.14.0 will produce lots of output)
    healpy
    pySLALIB (for coordinate transformations)
    matplotlib 1.3.1 or higher

Contents
--------
    psLLH       -   Pointsource Likelihood calculation and sensitivity
                    calculation
    ps_model    -   Likelihood models for use in psLLH
    ps_injector -   Source injectors for use in psLLH sensitivity estimation
    utils       -   Helping functions for other content that is not supported
                    in other libraries

"""

__author__ = "Stefan Coenders"
__version__ = "1.0"
__maintainer__ = "Stefan Coenders"
__email__ = "stefan.coenders@tum.de"
__status__ = "Development"

from glob import glob
import os

# Import PlotParser and all libraries in external python files
__all__ = [os.path.basename(f)[:-3]
            for f in glob(os.path.join(os.path.dirname(__file__), "*.py"))
            if not f.endswith("__init__.py")]
del f

def set_pars(self, **kwargs):
    r"""Constructor with basic settings needed in all LLH classes.

    """

    # Set all attributes passed, warn if private or not known
    for attr, value in kwargs.iteritems():
        if not hasattr(self, attr):
            print((">>>> {0:s} does not have attribute '{1:s}', "+
                   "skipping...").format(self.__repr__(), attr))
            continue
        if attr.startswith("_"):
            print((">>>> _{0:s} should be considered private and "+
                   "for internal use only!").format(attr))
        setattr(self, attr, value)

    return


