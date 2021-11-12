import sys
import pickle as pl
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

file_name  = sys.argv[1]

fig=pl.load(open(file_name,'rb'))
ax = fig.axes
#ax[1].set_ylim(top=5000000)
#ax[2].set_ylim(top=100000)
#ax[3].set_ylim(top=10000)
ax[1].set_ylim((-231273.82288284082, 5544749.334742298))
ax[2].set_ylim((-6656.804412145691, 159602.3391185104))
ax[3].set_ylim((-753.3520138754039, 18587.201570417976))

for i,x in enumerate(ax):
    x.set_title(x.get_title(),fontsize = 24)
    x.tick_params(axis='both', which='major', labelsize=16)
    x.set_xlim(right=datetime.date(day=28,month=10,year=2021))
    handles, labels = x.get_legend_handles_labels()
    labels[1]='With vaccines'
    x.legend(handles, labels, loc="upper left", fontsize = 16)
    x.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    x.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=2))
    ylabels = ['{:,.0f}'.format(y/1000) + 'K' if y>=1000 else '{:,.0f}'.format(y) for y in x.get_yticks()]
    x.set_yticklabels(ylabels)
    if i not in [1,2,3]:
        fig.delaxes(x)
for i,x in enumerate(fig.axes):
    x.change_geometry(1,3,i+1)

fig.set_size_inches((20,5), forward=False)
fig.savefig(file_name.split('.')[0]+'.png',dpi=600)
#pl.dump(fig, open(file_name, 'wb'))

plt.show()
