# -*-coding:utf8-*-

from distutils.core import setup

setup(name="skylab",
      version="1.0",
      author="Stefan Coenders",
      author_email="stefan.coenders@tum.de",
      packages=["skylab"],
      install_requires=["numpy>=1.9.0", "healpy", "scipy>=0.14"],
      license="GNU v3",
      )
