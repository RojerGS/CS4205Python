# %%
# setup
import pandas as pd
import numpy as np
import os
from launcher import population_sizes, ms, ks, crossoverTypes, time_limits

# create new columns in table
results = pd.read_csv('./experiments/results.csv', delimiter=',', header=0)
results['optimal_fraction'] = results.apply(lambda row: row.best_fitness / row.optimum, axis=1)
results['timed_out'] = results.apply(lambda row: row.time > time_limits[row.m], axis=1)

if 'results' not in os.listdir('experiments'):
    os.mkdir('experiments/results')





# %%
# plotting overhead stuff
import matplotlib.pyplot as plt

coNames = list(map(lambda x: x.name, crossoverTypes))
colors = dict(zip(coNames, ('r', 'b')))
pointColors = dict(zip(coNames, [(1, 0, 0, .25), (0, 0, 1, .25)]))
lineColors = dict(zip(coNames, [(1, 0, 0, .75), (0, 0, 1, .75)]))

k = ks[0]
dts = ['1/k', '1-1/k']
ds = [1/k, 1-1/k]

'''
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

cot0 = crossoverTypes[0]
cot1 = crossoverTypes[1]
legend_elements = [Line2D([0], [0], color=lineColors[cot0.name], lw=.5, label=cot0.name+" averages"),
                   Line2D([0], [0], color=lineColors[cot1.name], lw=.5, label=cot1.name+" averages"),
                   Line2D([0], [0], color='w', markerfacecolor=pointColors[cot0.name], marker='o', markersize=15, label=cot0.name+" point"),
                   Line2D([0], [0], color='w', markerfacecolor=pointColors[cot1.name], marker='o', markersize=15, label=cot1.name+" point")
                   ]

fig, ax = plt.subplots()
ax.legend(handles=legend_elements, loc='center')
plt.savefig('experiments/results/legend.png')
'''




# %% FITNESS
# plot fitness as a percentage of optimality by m value and d value
fn = "experiments/results/fitness.png"
f, axes = plt.subplots(len(ms), len(ds), sharex=True, sharey=True)
f.suptitle('Elite Fitness/Optimal by m, d')

for i in range(len(ms)):
    m = ms[i]

    for j in range(len(ds)):
        d = ds[j]

        for co in crossoverTypes:
            # collect data
            sub = results[(results.crossover_type == co.name)
                          & (results.m == m)
                          & (results.d == d)]
            averages = []
            for pop in sorted(set(sub.population_size)):
                averages.append((pop, np.mean(sub[sub.population_size == pop].optimal_fraction)))

            # plot scatter and line
            if i == len(ms)-1:
                axes[i][j].set_xlabel('d={0}\npopulation size'.format(d))
            if j == 0:
                axes[i][j].set_ylabel('m={0}'.format(m))

            axes[i][j].tick_params(labelsize=8)
            axes[i][j].set_xticks(list(range(0, 200+1, 20)))
            axes[i][j].scatter(sub.population_size, sub.optimal_fraction, color = pointColors[co.name])
            axes[i][j].plot(list(map(lambda x: x[0], averages)), list(map(lambda x: x[1], averages)), color = lineColors[co.name])

f.savefig(fn)





# %% CONVERGENCE
# plot the number of times our algorithms found the optimal genotype
for d in ds:
    fn = "experiments/results/converged{d}.png".format(d=str(d)[-1])
    f, axes = plt.subplots(len(ms), sharex=True)
    f.suptitle('Convergence by m for d={d}'.format(d=d))

    for i in range(len(ms)):
        m = ms[i]

        for j in range(len(crossoverTypes)):
            co = crossoverTypes[j]
            xs = np.array(population_sizes)

            axes[i].set_xticks(list(range(20, 120, 20)))
            axes[i].set_xlim(16, 120)
            axes[i].set_ylabel('m={m}'.format(m=m))

            # collect data
            sub = results[(results.crossover_type == co.name)
                          & (results.m == m)
                          & (results.d == d)]

            for p in population_sizes:
                optimalCount = len(sub[(sub.population_size == p)
                                        & (sub.is_optimal == True)])
                convergeCount = len(sub[(sub.population_size == p)
                                        & (sub.timed_out == False)])

                x = p
                if j == 1: x = x+8
                axes[i].bar(x, optimalCount, color=colors[co.name], hatch='.', width=4, align='edge')
                axes[i].bar(x+4, convergeCount, color=colors[co.name], hatch='\\', width=4, align='edge')

    f.savefig(fn)





# %% FINDING OPTIMAL SOLUTION
# plot number of generations needed to find optimal solution
fn = "experiments/results/convergenceTime.png".format(d=d)
f, axes = plt.subplots(len(ms), len(ds), sharex=True)
f.suptitle('Time Until Convergence by m, d'.format(d=d))

for i in range(len(ms)):
    m = ms[i]

    for j in range(len(ds)):
        d = ds[j]

        for co in crossoverTypes:
            # collect data
            sub = results[(results.crossover_type == co.name)
                          & (results.m == m)
                          & (results.d == d)
                          & (results.timed_out == False)
                          & (results.is_optimal == False)]
            averages = []
            for pop in sorted(set(sub.population_size)):
                averages.append((pop, np.mean(sub[sub.population_size == pop].time)))

            # plot scatter and line
            if i == len(ms)-1:
                axes[i][j].set_xlabel('d={0}\npopulation size'.format(d))
            if j == 0:
                axes[i][j].set_ylabel('m={0}'.format(m))

            axes[i][j].tick_params(labelsize=8)
            axes[i][j].set_xticks(list(range(0, 100+1, 20)))
            axes[i][j].scatter(sub.population_size, sub.time, color = pointColors[co.name])
            axes[i][j].plot(list(map(lambda x: x[0], averages)), list(map(lambda x: x[1], averages)), color = lineColors[co.name])

f.savefig(fn)
