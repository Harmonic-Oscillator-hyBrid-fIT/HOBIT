<p align="center">
 <img src="https://github.com/anamabo/HOBIT/blob/master/images/logo.png?raw=true" alt="Sublime's custom image"/>
</p>

# HOBIT: Harmonic Oscillator hyBrid fIT
## Efficient fit of sine(cosine) functions using a hybrid method

HOBIT is a Python library that combines the power of Hyperopt (https://github.com/hyperopt/hyperopt) with the flexibility of Sklearn oriented 
to the teaching of physics that is able to fit in a very efficient way functions with the shape

```
f(x) = y_0 + y_1 * Sin(omega * x + phi)
f(x) = y_0 + y_1 * Cos(omega * x + phi)
```

whose are commonly used to describe harmonic oscillators.

## Install
Install HOBIT 0.0.4
```
pip install HOBIT==0.0.4
```
### Requirements
Hobit will install/update the next python libraries

* pandas>=1.1.3
* numpy>=1.19.2
* hyperopt>=0.2.7
* sklearn>=0.23.2

## Get started
In the folder notebooks you will find four jupyther notebooks with the following examples:

1. Fit of a Cosine function using scipy and description of the gradient descent methond.
2. Fit of a Cosine function using Hyperopt package.
3. Fit of a Cosine function using HOBIT package.
4. Fit of a Cosine function using HOBIT package.

Detailed explanation on the usage of these codes are inside the notebooks.