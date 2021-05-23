# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.7 (tags/v3.7.7:d7c567b08f, Mar 10 2020, 10:41:24) [MSC v.1900 64 bit (AMD64)]
# Embedded file name: backprop_example.py
# Compiled at: 2021-04-29 11:35:24
# Size of source mod 2**32: 1859 bytes
import numpy as np

def backprop(X, y, theta, depth):
    J, yhat, h, a = forward_pass(X, y, theta, depth)
    gradients = backwards_pass(y, theta, h, a)
    return (
     J, yhat, gradients)


def forward_pass(X, y, theta, depth):
    h = {'h0': X}
    a = {}
    for k in range(depth):
        a['a' + str(k + 1)] = Dense(h[('h' + str(k))], theta[('w' + str(k + 1))], theta[('b' + str(k + 1))])
        if k != depth - 1:
            h['h' + str(k + 1)] = ReLU(a[('a' + str(k + 1))])
        else:
            h['h' + str(k + 1)] = Sigmoid(a[('a' + str(k + 1))])

    phat = h[('h' + str(depth))]
    J = BCE(y, phat)
    return (
     J, phat, h, a)


def backwards_pass(y, theta, h, a):
    gradients = {}
    g = dBCE_dphat(y, h[('h' + str(len(h) - 1))])
    for k in range(len(a), 0, -1):
        if k != len(a):
            g = g * dReLU_da(a[('a' + str(k))])
        gradients['dJ_db' + str(k)] = g @ np.ones(y.T.shape)
        gradients['dJ_dw' + str(k)] = g @ h[('h' + str(k - 1))].T
        g = theta[('w' + str(k))].T @ g

    return gradients


def MSE(y, yhat):
    mse = np.mean((y - yhat) ** 2)
    return mse


def BCE(y, phat):
    bce = -np.mean(y * np.log(phat) + (1 - y) * np.log(1 - phat))
    return bce


def dMSE_dyhat(y, yhat):
    dmse_dyhat = -2 / np.size(yhat) * (y - yhat)
    return dmse_dyhat


def dBCE_dphat(y, phat):
    dbce_dphat = 1 / np.size(phat) * -(y / phat - (1 - y) / (1 - phat))
    return dbce_dphat


def dReLU_da(a):
    dh_da = 1 * (a > 0)
    return dh_da


def dSigmoid_da(a):
    dh_da = Sigmoid(a) * (1 - Sigmoid(a))
    return dh_da


def Dense(h, w, b):
    a = w @ h + b
    return a


def ReLU(a):
    h = np.maximum(0, a)
    return h


def Sigmoid(a):
    h = 1 / (1 + np.exp(-a))
    return h