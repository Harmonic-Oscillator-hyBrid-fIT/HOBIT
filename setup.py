from setuptools import setup, Extension


with open("requirements.txt") as f:
    requirements = f.read().splitlines()
with open("README.md") as f:
    long_description = f.read()

setup(
    name='HOBIT',
    version='0.0.4',
    description='an installable package for Hybrid fitting of Sine and Cosine functions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Harmonic-Oscillator-hyBrid-fIT/HOBIT',
    author='Carmen Adriana Martinez Barbosa, Jose Arturo Celis Gil',
    author_email='anamabo3@gmail.com, solocelis@gmail.com',
    packages=['HOBIT'],
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    zip_safe=False
)
