# -*-coding:utf8-*-

import os
from distutils.core import setup

with open(os.path.join(os.path.dirname(__file__), "README.md")) as of:
    desc = "\n".join(of.readlines())

setup(name="skylab",
      version="1.0",
      description="Skylab - unbinned likelihood clustering analyses in the sky",
      long_description=desc,
      author="Stefan Coenders",
      author_email="stefan.coenders@tum.de",
      download_url="http://github.com/coenders/skylab",
      packages=["skylab"],
      install_requires=["numpy>=1.9.0", "healpy", "scipy>=0.14"],
      license="GNU v3",
      )
