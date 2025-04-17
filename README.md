# Learning to Compose Quaternions Without a Group

All source files are located in `src/`.

- `train.py` – trains the ODE-RNN on sequences of unit quaternions
- `interp.py` – computes Jacobians of the learned vector field at each step
- `trajectory.py` – collects latent trajectories and compares them to SLERP curves
- `plot_jacobians.py` – visualises Jacobian statistics over time
- `plot_trajectory.py` – visualises geodesic deviation and norm drift
- other files define the model, loss functions, and utility code

## Install
```bash
pip install -r requirements.txt
```