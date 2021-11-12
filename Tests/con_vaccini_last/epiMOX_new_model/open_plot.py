import sys
import pickle as pl
import matplotlib
import matplotlib.pyplot as plt

file_name  = sys.argv[1]

fig=pl.load(open(file_name,'rb'))
plt.show()
