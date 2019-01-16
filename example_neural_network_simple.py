#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:50:08 2019

@author: abhisheklomsh

Here I wrote this piece to solve an example on nueral network as part of a course on coursera
"""

import numpy as np

sigma=np.tanh
w1=1.3
b1=-0.1

def a1(a0):
    return sigma(w1*a0+b1)

val=a1(3)
print(val)