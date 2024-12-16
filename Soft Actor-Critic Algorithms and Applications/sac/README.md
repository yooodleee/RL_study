## **1. Install `mujoco_py`**
First, install the `mujoco_py` package:

```bash
pip install mujoco-py
```

If you encounter issues during installation, follow the additional steps below to properly set up the environment.

---

## **2. Install the MuJoCo Library**
`mujoco-py` requires the MuJoCo library. You'll need to install and set it up.

### **(1) Download MuJoCo**
You can download the MuJoCo library from the [official MuJoCo website](https://mujoco.org/download).

1. **Download the latest version** from the provided link.
2. **Extract the downloaded file** to a proper directory, e.g., `C:\Users\<username>\.mujoco` (on Windows).

---

### **(2) Set Up Environment Variables**
After installing MuJoCo, you need to configure the environment variables.

1. **Open Environment Variables**
   - Windows: Press `Win + R` → Type `sysdm.cpl` → Go to the "Advanced" tab → Click "Environment Variables."

2. **Add New Variables:**
   - **New variable for the MuJoCo key**:
     - Variable name: `MUJOCO_PY_MJKEY_PATH`
     - Variable value: `<MuJoCo installation path>\mjkey.txt`
   - **New variable for the MuJoCo directory**:
     - Variable name: `MUJOCO_PY_MJPRO_PATH`
     - Variable value: `<MuJoCo installation path>\mujoco210`

3. **Add MuJoCo to the system Path:**
   - Add `<MuJoCo installation path>\bin` to your system Path environment variable.

---

## **3. Install Additional Dependencies**
`mujoco-py` requires `Cython` and `numpy`. Make sure they are installed:

```bash
pip install numpy cython
```

---

## **4. Test the `mujoco-py` Installation**
After installation, test if the `mujoco-py` package is correctly installed by running the following in Python:

```python
import mujoco_py
print(mujoco_py.__file__)
```

If the file path is printed without errors, the installation is successful.

---

## **5. Gym Version Compatibility**
The latest version of `gym` might support the `mujoco` package instead of `mujoco-py`.  
- You can install `mujoco` via `pip install mujoco`.
- Consider switching from `mujoco-py` to `mujoco` in your code if you're using the latest `gym`.

---

## **6. Troubleshooting**
If you still encounter issues after completing the steps above:
1. **Permissions Check**: Ensure that the MuJoCo directory is readable and writable.
2. **PyEnv Virtual Environment**: Verify that the package is correctly installed within the `pyenv` environment you're using.
3. **Share Logs**: If the error persists, sharing the specific error logs will help in providing more targeted support.

---

## **7. About Papers**
Model-free deep reinforcement learning(RL) algorithms have been successfully applied to a range of challenging sequential decision making and control tasks.
However, these methods typically suffer from two major challenges:

1. High sample complexity
2. Brittleness to hyperparameters.

Both ot these challenges limit the applicability of such methods to real-world domains.

Soft Acotr-Critic (SAC), recently introduced off-policy actor-critic algorithm based on the maximum entropy RL framework. 
In this framework, the actor aims to simultaneously maximize expected return and entropy; that is, to suceed at the task while acting as randomly as possible.

SAC achieves state-of-the-art performance, ourtperforming prior on-policy and off-policy methods in sample-efficiency and asymptotic performance.
Furthermore, in contrast to other off-policy algorithms, achieving similar performance across different random seeds.
These results suggest that SAC is a promising candidate for learning in real-world robotics tasks.

---

## **8. Hyperparameters**
SAC Hyperparameters

1. `optimizer`: Adam
2. `learning rate`: 3.10^-4
3. `discount factor(𝛾)`: 0.99
4. `replacy buffer size`: 10^6
5. `number of hidden layers (all networks)`: 2
6. `number of hidden units per layer`: 256
7. `number of samples per minibatch`: 256
8. `entropy target`: -dim (A) (e.g., -6 for HalfCheetah-v1)
9. `nonlinearity`: ReLU
10. `target smoothing coefficient (𝜏)`: 0.005
11. `target update interval`: 1
12. `gradient steps`: 1
