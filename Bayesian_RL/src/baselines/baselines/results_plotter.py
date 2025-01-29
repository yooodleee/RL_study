import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

import matplotlib.pyplot as plt
plt.reParams['svg.fonttype'] = 'none'

from baselines.bench.monitor import load_results


X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
Y_REWARD = 'reward'
Y_TIMESTEPS = 'timesteps'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = [
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 
    'black', 'purple', 'pink', 'brown', 'orange', 'teal', 
    'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
    'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 
    'darkred', 'darkblue'
]


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def ts2xy(ts, xaxis, yaxis):
    if xaxis == X_TIMESTEPS:
        x = np.cumsum(ts.l.values)
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
    else:
        raise NotImplementedError
    
    if yaxis == Y_REWARD:
        y = ts.r.values
    elif yaxis == Y_TIMESTEPS:
        y = ts.l.values
    else:
        raise NotImplementedError
    
    return x, y


def plot_curves(xy_list, xaxis, yaxis, title):
    fig = plt.figure(figsize=(8, 2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i]
        plt.scatter(x, y, s=2)
        x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean) # So returns average of last EPISODE_WINDOW episodes
        plt.plot(x, y_mean, color=color)
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout()
    fig.canvas.mpl_connect('resize_event', lambda event: plt.tight_layout())
    plt.grid(True)


