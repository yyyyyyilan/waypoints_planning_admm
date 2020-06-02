from numpy import *
from scipy.sparse.linalg import expm
from scipy.linalg import norm
import pdb
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.io import loadmat
import numpy as np
import random

def traj_mwpts(t, b, vi, ai, zi):
    # Optimal trajectory passing thru multiple waypoints
    # Continuous dynamics: \dot{x} = Ax + Bu; y = Cx;
    m = 4.34
    A = np.array(np.vstack((np.hstack((np.zeros((3,3)), np.eye(3), np.zeros((3,3)), np.zeros((3,3)))),
                            np.hstack((np.zeros((3,3)), np.zeros((3,3)), np.multiply(np.divide(-1., m), np.eye(3)), np.zeros((3,3)))),
                            np.hstack((np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.eye(3))),
                            np.hstack((np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)))))))
    B = np.array(np.hstack((np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.eye(3)))).T
    C = np.array(np.hstack((np.eye(3), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)))))

    # Discrete dynamics: x_{i+1} = Adx_{i} + Bdu_{i}; y_{i} = Cdu_{i};
    h = 0.1
    Ad = expm(A*h)
    Bd = np.array(np.vstack((np.hstack((h*np.eye(3), ((pow(h,2))/2)*np.eye(3), ((pow(h,3))/6)*np.eye(3), ((pow(h,4))/24)*np.eye(3))),
                            np.hstack((np.zeros((3, 3)), h*np.eye(3), ((pow(h,2))/2)*np.eye(3), ((pow(h,3))/6)*np.eye(3))),
                            np.hstack((np.zeros((3, 3)), np.zeros((3, 3)),  h*np.eye(3), ((pow(h,2))/2)*np.eye(3))),
                            np.hstack((np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), h*np.eye(3))))))
    Bd = Bd.dot(B)
    Cd = C.copy()

    # Physical parameters
    g = 9.8 # Acceleration due to gravity
    e3 = array([[0],[0],[1]])

    # Initial state
    xi = np.vstack((np.expand_dims(b[:,0], axis=1), vi, ai, zi))

    # Gain Matrices
    R_bar = 25 * eye(3)
    Q_bar = 0.01 * eye(12)
    S = 35 * np.array([[2, 0, 0], [0, 2, 0], [0, 0, 10]])

    Pf = Cd.T.dot(S).dot(C)/h
    etaf =  -1*Cd.T.dot(S).dot(np.expand_dims(b[:,-1], axis=1))/h

    N = int(np.round((t[-1] - t[0])/h))
    nm = zeros(len(t), dtype=np.int)
    for i in range(len(t)-1):
        nm[i+1] = nm[i] + int(np.round((t[i+1]-t[i])/h))

    P = np.zeros((12, 12, N+1))
    P[:, :, N] = Pf
    eta = np.zeros((12, N+1))
    eta[:, N] = etaf.squeeze()
    k1 = 10

    R = np.zeros((3, 3, N))
    Q = np.zeros((12, 12, N))
    for j in range(len(nm)-1, 0, -1):
        alfa = np.zeros((1, nm[j]-nm[j-1]-1))
        for k in range(int(nm[j]-nm[j-1]-1)):
            alfa[0][k] = ((k+1)*h)/(t[j]-t[j-1])

        for i in range(nm[j]-1, nm[j-1], -1):
            R[:,:,i] = (k1-alfa[0][i-1-nm[j-1]]) * R_bar
            Q[:,:,i] = alfa[0][i-1-nm[j-1]] * Q_bar
            P[:,:,i] = Q[:,:,i]+Ad.T.dot(P[:,:,i+1]).dot(Ad)-Ad.T.dot(P[:,:,i+1]).dot(Bd).dot(np.linalg.inv(R[:,:,i]+Bd.T.dot(P[:,:,i+1]).dot(Bd))).dot(Bd.T).dot(P[:,:,i+1]).dot(Ad)
            eta[:,i] = Ad.T.dot(np.eye(12)-P[:,:,i+1].dot(Bd).dot(np.linalg.inv(R[:,:,i]+Bd.T.dot(P[:,:,i+1]).dot(Bd))).dot(Bd.T)).dot(eta[:,i+1])

        k = nm[j-1]
        alfa = 1
        R[:,:,k] = (k1-alfa)*R_bar
        Q[:,:,k] = alfa*Q_bar
        P[:,:,k] = Q[:,:,k]+Ad.T.dot(P[:,:,k+1]).dot(Ad)-Ad.T.dot(P[:,:,k+1]).dot(Bd).dot(np.linalg.inv(R[:,:,k]+Bd.T.dot(P[:,:,k+1]).dot(Bd))).dot(Bd.T).dot(P[:,:,k+1]).dot(Ad)+Cd.T.dot(S).dot(Cd)/h
        eta[:, k] = Ad.T.dot(np.eye(12)-P[:,:,k+1].dot(Bd).dot(np.linalg.inv(R[:,:,k] + Bd.T.dot(P[:,:,k+1]).dot(Bd))).dot(Bd.T)).dot(eta[:,k+1]) - (Cd.T.dot(S).dot(np.expand_dims(b[:,j-1], axis=1)/h)).squeeze()

    # Determining state and control
    xt = np.zeros((12, N))
    xt[:,0] = xi.squeeze()
    tme = np.zeros((1,N))
    tme[:,0] = t[0]
    thrst = np.zeros((3, N))
    thrst[:,0] = (m * (g * e3 + np.expand_dims(xt[6:9, 0], axis=1))).squeeze()
    nthr = np.zeros((1, N))
    nthr[:,0] = norm(thrst[:, 0])
    spd = np.zeros((1, N))
    spd[:, 0] = norm(xt[3:6, 0])

    ut = np.zeros((3, N-1))
    yt = np.zeros((3, N))
    for j in range(N-1):
        tme[:,j+1] = t[0]+(j+1)*h
        ut[:,j] =  -1*np.linalg.inv(R[:,:,j]+Bd.T.dot(P[:,:,j+1]).dot(Bd)).dot(Bd.T).dot(P[:,:,j+1].dot(Ad).dot(xt[:,j])+eta[:,j+1])
        xt[:,j+1] = Ad.dot(xt[:,j])+Bd.dot(ut[:,j])
        yt[:,j+1] = Cd.dot(xt[:,j+1])
        thrst[:,j+1] = (m * (g * e3 + np.expand_dims(xt[6:9, j+1], axis=1))).squeeze()
        nthr[:,j+1] = norm(thrst[:,j+1])
        spd[:,j+1] = norm(xt[3:6, 0])

    lamb = np.zeros((12, N))
    for j in range(N):
        lamb[:, j] = P[:,:,j].dot(xt[:,j])+eta[:,j]

    return [xt[0:3, :], thrst, nthr, xt[3:6, -1], xt[6:9, -1], xt[9:12, -1]]
