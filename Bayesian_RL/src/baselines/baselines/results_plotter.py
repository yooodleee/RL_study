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


