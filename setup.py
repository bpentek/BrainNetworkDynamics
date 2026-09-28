from setuptools import setup, Extension
import os
from Cython.Build import cythonize
import numpy as np

extensions = [  Extension(  name="brainnetworkdynamics.sca.sca_cy", 
                            sources=[os.path.join("src", "brainnetworkdynamics", "sca", "sca_cy.pyx")],
                            include_dirs=[np.get_include()])    ]

setup(  name="brainnetworkdynamics",
        version="1.0.0",
        ext_modules=cythonize(extensions),
        include_dirs=[np.get_include()],
        options={
            'bdist_wheel': {
                'universal':True,
            },
        },
      )
