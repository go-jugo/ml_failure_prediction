# -*- coding: utf-8 -*-
"""
Module that creates the global signature 
-> necessary because we want to use and change the signature in different modules(main.py und time_it.py)
"""

class Signature():
    def __init__(self,sig=''):
        self.sig = sig

GLOBAL_SIGNATURE = Signature()