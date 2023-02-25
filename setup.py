from setuptools import setup, find_packages

VERSION = '0.0.2' 
DESCRIPTION = 'Diffusion model for galaxy generation'
LONG_DESCRIPTION = 'Contains two jax functions ScoreNet32 and ScoreNet64. These are used to return the gradients of an arbitrary image with respect to a prior distribution of individual artifact free galaxy models. Current functions include ScoreNetXX(image) returns gradients as stated. generateSample(samples=n, hi_res=bool, seed=XXXX) will generate an array of n galaxy samples which can be plotted with imshow.'

# Setting up
setup(
        name="galaxygrad", 
        version=VERSION,
        author="Matt Sampson",
        author_email="matt.sampson@princeton.edu",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        package_data={'':['*.eqx']},
        include_package_data=True,
        install_requires=['jax', 'equinox', 'einops', 'diffrax','functools'],
        
        keywords=['python', 'diffusion'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
        ]
)
