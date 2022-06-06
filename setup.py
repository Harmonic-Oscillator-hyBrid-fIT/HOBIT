from setuptools import setup, Extension


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name='HOBIT',
    version='0.0.3',
    description='an installable package for Hybrid fitting of Sine and Cosine functions',
    url='https://github.com/Harmonic-Oscillator-hyBrid-fIT/HOBIT',
    author='Carmen Adriana Martinez Barbosa, Jose Arturo Celis Gil',
    author_email='anamabo3@gmail.com, solocelis@gmail.com',
    packages=['HOBIT'],
    install_requires=requirements,
    classifiers=[
        "Programmiing Language :: Python :: 3",
        "Operatng System :: OS IIndependent"
    ],
    zip_safe=False
)
