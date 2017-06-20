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
======

SkyLab is a project for unbinned likelihood analyses of point source
searches as performed by Neutrino telescopes like ANTARES or IceCube.

Requirements
------------

* healpy
* numpy 1.8 or higher
* scipy 0.14.0 or higher

Contents
--------

basellh
    Base class for likelihood implementations; provides methods for
    sensitivity calculation

psLLH
    Likelihood implementations for single or multi-year analyses

grbllh
    Likelihood implementation for gamma-ray burst analyses

ps_model
    Likelihood models

ps_injector
    Source injectors for sensitivity caclulation

utils
    Helping functions

"""
import logging

__all__ = ["basellh", "psLLH", "grbllh", "ps_model", "ps_injector", "utils"]

__author__ = "Stefan Coenders"
__version__ = "1.0"
__maintainer__ = "Stefan Coenders"
__email__ = "stefan.coenders@tum.de"
__status__ = "Development"

logging.getLogger(__name__).addHandler(logging.NullHandler())
