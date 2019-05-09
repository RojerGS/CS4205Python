# %%
# setup
import pandas as pd
import numpy as np
import os
from launcher import population_sizes, ms, ks, crossoverTypes

list(map(lambda x: x.name, crossoverTypes))

results = pd.read_csv('./experiments/results.csv', delimiter=',', header=0)
results['optimal_fraction'] = results.apply(lambda row: row.best_fitness / row.optimal, axis=1)

if 'results' not in os.listdir('experiments'):
    os.mkdir('experiments/results')





# %%
# plotting overhead stuff
pointColors = dict(zip(list(map(lambda x: x.name, crossoverTypes)), [(1, 0, 0, .25), (0, 0, 1, .25)]))
lineColors = dict(zip(list(map(lambda x: x.name, crossoverTypes)), [(1, 0, 0, .75), (0, 0, 1, .75)]))

import matplotlib.pyplot as plt
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

k = ks[0]
dts = ['1/k', '1-1/k']
ds = [1/k, 1-1/k]




""" OPTIMALITY """
# %%
# plot percentage of optimality by m value
fn = "experiments/results/optimalityByM.png"
f, axes = plt.subplots(len(ms), sharex=True)
f.suptitle('Elite Fitness Optimality by m')
axes[len(ms)-1].set_xlabel('population size')
axes[len(ms)-1].set_xticks (list(range(0, 200+1, 20)))

for i in range(len(ms)):
    m = ms[i]
    for co in crossoverTypes:
        # collect data
        sub = results[(results.crossover_type == co.name) & (results.m == m)]
        averages = []
        for pop in sorted(set(sub.population_size)):
            averages.append((pop, np.mean(sub[sub.population_size == pop].optimal_fraction)))

        # plot scatter and line
        axes[i].set_ylabel('m='+str(m))
        axes[i].scatter(sub.population_size, sub.optimal_fraction, color = pointColors[co.name])
        axes[i].plot(list(map(lambda x: x[0], averages)), list(map(lambda x: x[1], averages)), color = lineColors[co.name])

f.savefig(fn)

# %%
# plot percentage of optimality by d value
fn = "experiments/results/optimalityByD.png"
f, axes = plt.subplots(len(ds), sharex=True)
f.suptitle('Elite Fitness Optimality by d')
axes[len(ds)-1].set_xlabel('population size')
axes[len(ds)-1].set_xticks (list(range(0, 200+1, 20)))

for co in crossoverTypes:
    for i in range(len(dts)):
        d = ds[i]
        # collect data
        sub = results[(results.crossover_type == co.name) & (results.d == d)]
        averages = []
        for pop in sorted(set(sub.population_size)):
            averages.append((pop, np.mean(sub[sub.population_size == pop].optimal_fraction)))

        # plot scatter and line
        axes[i].set_ylabel('d = '+str(d))
        axes[i].scatter(sub.population_size, sub.optimal_fraction, color = pointColors[co.name])
        axes[i].plot(list(map(lambda x: x[0], averages)), list(map(lambda x: x[1], averages)), color = lineColors[co.name])

f.savefig(fn)





""" EVALUATIONS """
# %%
# plot percentage of optimality by m value
fn = "experiments/results/evaluationsByM.png"
f, axes = plt.subplots(len(ms), sharex=True)
f.suptitle('Number of Evaluations by m')
axes[len(ms)-1].set_xlabel('population size')
axes[len(ms)-1].set_xticks (list(range(0, 200+1, 20)))

for i in range(len(ms)):
    m = ms[i]
    for co in crossoverTypes:
        # collect data
        sub = results[(results.crossover_type == co.name) & (results.m == m) & (results.isOptimal == True)]
        averages = []
        for pop in sorted(set(sub.population_size)):
            averages.append((pop, np.mean(sub[sub.population_size == pop].evaluations)))

        # plot scatter and line
        axes[i].set_ylabel('m='+str(m))
        axes[i].scatter(sub.population_size, sub.evaluations, color = pointColors[co.name])
        axes[i].plot(list(map(lambda x: x[0], averages)), list(map(lambda x: x[1], averages)), color = lineColors[co.name])

f.savefig(fn)

# %%
# number of evaluations needed to reach optimality by d value
fn = "experiments/results/evaluationsByD.png"
f, axes = plt.subplots(len(ds), sharex=True)
f.suptitle('Number of Evaluations by d')
axes[len(ds)-1].set_xlabel('population size')
axes[len(ds)-1].set_xticks (list(range(0, 200+1, 20)))

for co in crossoverTypes:
    for i in range(len(dts)):
        d = ds[i]
        # collect data
        sub = results[(results.crossover_type == co.name) & (results.d == d) & (results.isOptimal == True)]
        averages = []
        for pop in sorted(set(sub.population_size)):
            averages.append((pop, np.mean(sub[sub.population_size == pop].evaluations)))

        # plot scatter and line
        axes[i].set_ylabel('d = '+str(d))
        axes[i].scatter(sub.population_size, sub.evaluations, color = pointColors[co.name])
        axes[i].plot(list(map(lambda x: x[0], averages)), list(map(lambda x: x[1], averages)), color = lineColors[co.name])

f.savefig(fn)





""" GENERATIONS """
# %%
# plot percentage of optimality by m value
fn = "experiments/results/generationsByM.png"
f, axes = plt.subplots(len(ms), sharex=True)
f.suptitle('Number of Generations by m')
axes[len(ms)-1].set_xlabel('population size')
axes[len(ms)-1].set_xticks (list(range(0, 200+1, 20)))

for i in range(len(ms)):
    m = ms[i]
    for co in crossoverTypes:
        # collect data
        sub = results[(results.crossover_type == co.name) & (results.m == m) & (results.isOptimal == True)]
        averages = []
        for pop in sorted(set(sub.population_size)):
            averages.append((pop, np.mean(sub[sub.population_size == pop].generations)))

        # plot scatter and line
        axes[i].set_ylabel('m='+str(m))
        axes[i].scatter(sub.population_size, sub.generations, color = pointColors[co.name])
        axes[i].plot(list(map(lambda x: x[0], averages)), list(map(lambda x: x[1], averages)), color = lineColors[co.name])

f.savefig(fn)

# %%
# number of evaluations needed to reach optimality by d value
fn = "experiments/results/generationsByD.png"
f, axes = plt.subplots(len(ds), sharex=True)
f.suptitle('Number of Generations by d')
axes[len(ds)-1].set_xlabel('population size')
axes[len(ds)-1].set_xticks (list(range(0, 200+1, 20)))

for co in crossoverTypes:
    for i in range(len(dts)):
        d = ds[i]
        # collect data
        sub = results[(results.crossover_type == co.name) & (results.d == d) & (results.isOptimal == True)]
        averages = []
        for pop in sorted(set(sub.population_size)):
            averages.append((pop, np.mean(sub[sub.population_size == pop].generations)))

        # plot scatter and line
        axes[i].set_ylabel('d = '+str(d))
        axes[i].scatter(sub.population_size, sub.generations, color = pointColors[co.name])
        axes[i].plot(list(map(lambda x: x[0], averages)), list(map(lambda x: x[1], averages)), color = lineColors[co.name])

f.savefig(fn)
