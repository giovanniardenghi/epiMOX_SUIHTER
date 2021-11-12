import numpy as np
import matplotlib.pyplot as plt
import sys

fileName = sys.argv[1]

R = np.loadtxt(fileName)

k = np.arange(10)/10
b = np.arange(15,30)/10
X,Y = np.meshgrid(k,b)

fig,ax = plt.subplots()

cont = ax.contourf(X,Y,R,levels=[0,1,2],cmap='RdYlGn_r')
CS = ax.contour(X,Y,R,levels=[0,1,2],colors='black')
fmt = {1:'Rt = 1'}
ax.clabel(CS, [1], inline=True, fmt=fmt, fontsize=15, colors='black')
ax.set_xlabel(r'$\kappa$',fontsize=20)
ax.set_ylabel(r'${\beta_v}/{\beta}$',fontsize=20)
plt.show()
