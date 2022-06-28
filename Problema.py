# -*- coding: utf-8 -*-
import ModeloSTHE as mSTHE
import CosteoSTHE as cSTHE
import numpy as np


def ProblemaOpt(x):
    AreaSTHE, dPTubos, dPTotCoraza, VelFluTubos, Ltubos, Fm_s, Fm_t, L_D = mSTHE.STHEFullModel(
        x)
    CostoGlob = cSTHE.CostoSTHE(AreaSTHE, dPTubos, dPTotCoraza, VelFluTubos, Fm_s, L_D, Fm_t)
    return CostoGlob
