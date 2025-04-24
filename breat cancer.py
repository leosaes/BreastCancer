# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 16:49:13 2025

@author: leosa
"""

import numpy as np
from sklearn import datasets

def sigmoid(soma):
    return 1/(1+np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1-sig)

base = datasets.load_breast_cancer()
entradas = base.data
valoresSaida = base.target
saidas = np.empty([569,1], dtype=int)
for i in range(569):
    saidas[i] = valoresSaida[i]


pesos0 = 2*np.random.random((30,5)) - 1
pesos1 = 2*np.random.random((5,1)) - 1

epocas = 10
taxaApr = 0.5
momento = 1

for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada,pesos0) 
    camadaOculta = sigmoid(somaSinapse0)
    somaSinapse1 = np.dot(camadaOculta,pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erro = saidas - camadaSaida
    mediaAbs = np.mean(np.abs(erro))
    print("Erro: " + str(mediaAbs))
    
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erro * derivadaSaida
    
    pesos1Transposta = pesos1.T
    deltaSaidaxPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaxPeso * sigmoidDerivada(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovos = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovos * taxaApr)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovosEntrada = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento)+(pesosNovosEntrada*taxaApr)