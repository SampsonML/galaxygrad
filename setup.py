from setuptools import setup, find_packages

VERSION = '0.0.19' 
DESCRIPTION = 'Diffusion model for galaxy generation'
LONG_DESCRIPTION = 'Contains two jax functions ScoreNet32 and ScoreNet64. These are used to return the gradients of an arbitrary image with respect to a prior distribution of individual artifact free galaxy models. '

# Setting up
setup(
        name="galaxynet", 
        version=VERSION,
        author="Matt Sampson",
        author_email="matt.sampson@princeton.edu",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        package_data={'':['*.eqx']},
        include_package_data=True,
        install_requires=[], # add  req ie jax, equinox
        
        keywords=['python', 'diffusion'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
