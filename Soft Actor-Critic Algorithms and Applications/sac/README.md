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

Feel free to ask if you need further assistance! 😊