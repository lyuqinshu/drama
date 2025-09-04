from scipy.linalg import sqrtm
import numpy as np
import os
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def load_force_results_bv(output_dir="results/force_bv", include_samples=False):
    """
    Load the results saved by `compute_forces_bv_to_files`, including sratio and Bx from filename.

    Parameters
    ----------
    output_dir : str
        Directory containing the saved .npz files.
    include_samples : bool
        If True, also return raw phase and force sample arrays for each point.

    Returns
    -------
    results : dict
        Dictionary with arrays of all parameters:
        - 'B', 'v', 'delta', 'Delta', 's', 'sratio', 'Bx'
        - 'mean_force', 'err'
        - optionally: 'forces_list', 'phases_list'
    """
    Bs, vs, deltas, Deltas, ss, sratios, Bxs = [], [], [], [], [], [], []
    mean_forces, errs = [], []
    forces_list, phases_list = [], []

    for fname in sorted(os.listdir(output_dir)):
        if not (fname.endswith(".npz") and fname.startswith("bv_b")):
            continue

        # Remove extension
        fname_base = fname[:-4]

        # Split by _
        parts = fname_base.split("_")[1:]
        params = {}
        for part in parts:
            if part.startswith("b"):
                params['B'] = float(part[1:].replace('p','.'))
            elif part.startswith("v"):
                params['v'] = float(part[1:].replace('p','.'))
            elif part.startswith("d"):
                params['delta'] = float(part[1:].replace('p','.'))
            elif part.startswith("D"):
                params['Delta'] = float(part[1:].replace('p','.'))
            elif part.startswith("sratio"):
                params['sratio'] = float(part[6:].replace('p','.'))
            elif part.startswith("Bx"):
                params['Bx'] = float(part[2:].replace('p','.'))
            elif part.startswith("s"):
                params['s'] = float(part[1:].replace('p','.'))

        # Load file
        path = os.path.join(output_dir, fname)
        data = np.load(path)

        # Append parsed and loaded values
        Bs.append(params.get('B'))
        vs.append(params.get('v'))
        deltas.append(params.get('delta'))
        Deltas.append(params.get('Delta'))
        ss.append(params.get('s'))
        sratios.append(params.get('sratio'))
        Bxs.append(params.get('Bx'))

        mean_forces.append(data["mean_force"].item())
        errs.append(data["err"].item())

        if include_samples:
            forces_list.append(data["forces"])
            phases_list.append(data["phases"])

    results = {
        "B": np.array(Bs),
        "v": np.array(vs),
        "delta": np.array(deltas),
        "Delta": np.array(Deltas),
        "s": np.array(ss),
        "sratio": np.array(sratios),
        "Bx": np.array(Bxs),
        "mean_force": np.array(mean_forces),
        "err": np.array(errs)
    }

    if include_samples:
        results["forces_list"] = forces_list
        results["phases_list"] = phases_list

    return results



def load_population_results(output_dir="data"):
    """
    Load population results saved by `evolve_dens`.

    Parameters
    ----------
    output_dir : str
        Directory containing the saved .npz files.

    Returns
    -------
    results : dict
        Dictionary with arrays of all parameters:
        - 'B', 'v', 'd', 'D', 's'
        - 'rho00', 'rho11', 'rho22', 'excited'
    """
    Bs, vs, ds, Ds, ss = [], [], [], [], []
    rho00_mean, rho11_mean, rho22_mean, excited_mean = [], [], [], []
    rho00_err, rho11_err, rho22_err, excited_err = [], [], [], []

    for fname in sorted(os.listdir(output_dir)):
        if not (fname.endswith(".npz") and fname.startswith("pop")):
            continue
        

        path = os.path.join(output_dir, fname)
        data = np.load(path)

        Bs.append(data['b'])
        vs.append(data['v'])
        ds.append(data['delta'])
        Ds.append(data['Delta'])
        ss.append(data['s'])

        rho00_mean.append(data["mean_rho00"])
        rho00_err.append(data["err_rho00"])
        rho11_mean.append(data["mean_rho11"])
        rho11_err.append(data["err_rho11"])
        rho22_mean.append(data["mean_rho22"])
        rho22_err.append(data["err_rho22"])
        excited_mean.append(data["mean_exc"])
        excited_err.append(data["err_excited"])

    results = {
        "B": np.array(Bs),
        "v": np.array(vs),
        "delta": np.array(ds),
        "Delta": np.array(Ds),
        "s": np.array(ss),
        "rho00": np.array(rho00_mean),
        'rho00_err': np.array(rho00_err),
        "rho11": np.array(rho11_mean),
        'rho11_err': np.array(rho11_err),
        "rho22": np.array(rho22_mean),
        'rho22_err': np.array(rho22_err),
        "excited": np.array(excited_mean),
        'excited_err': np.array(excited_err),
    }

    return results

def read_bv_force_results(directory: str):
    """
    Reads all .npz result files generated by compute_forces_bv_to_files
    from the specified directory and returns a structured NumPy array.

    Parameters
    ----------
    directory : str
        Path to the folder containing result .npz files.

    Returns
    -------
    np.ndarray
        Structured NumPy array with fields:
        'b', 'v', 'mean_force', 'err', 'forces', 'phases'
    """
    records = []

    for fname in sorted(os.listdir(directory)):
        if fname.endswith('.npz'):
            filepath = os.path.join(directory, fname)
            npz = np.load(filepath)

            records.append((
                npz['b'].item(),
                npz['v'].item(),
                npz['mean_force'].item(),
                npz['err'].item(),
                npz['forces'],
                npz['phases']
            ))

    dtype = np.dtype([
        ('b', 'f8'),
        ('v', 'f8'),
        ('mean_force', 'f8'),
        ('err', 'f8'),
        ('forces', 'O'),  # Object for variable-length array
        ('phases', 'O')   # Object for variable-length 2D array
    ])

    return np.array(records, dtype=dtype)


def fidelity(rho1, rho2):
    """
    Compute the quantum fidelity between two density matrices.

    Parameters
    ----------
    rho1, rho2 : np.ndarray
        2D complex density matrices of the same shape.

    Returns
    -------
    float
        Fidelity value between 0 and 1.
    """
    sqrt_rho1 = sqrtm(rho1)
    inner_term = sqrtm(sqrt_rho1 @ rho2 @ sqrt_rho1)
    return np.real(np.trace(inner_term)) ** 2

def plot_density_matrix_fidelity(res, res_2):
    """
    Given two result objects with .t and .rho (6x6xT), compute and plot fidelity over time.

    Parameters
    ----------
    res : object
        Object with attributes `.t` (array of times) and `.rho` (6x6xT density matrix).
    res_2 : object
        Second result object to compare against (possibly different time sampling).

    Returns
    -------
    list of float
        Fidelity values at each time point in res.t.
    """
    # Reshape rho to shape (T, 6, 6)
    rho1_list = np.transpose(res.rho, (2, 0, 1))     # (N, 6, 6)
    rho2_list = np.transpose(res_2.rho, (2, 0, 1))   # (M, 6, 6)

    # Flatten rho2 for interpolation
    rho2_flat = rho2_list.reshape((len(res_2.t), -1))  # (M, 36)

    # Interpolate each element over res.t
    interp_func = interp1d(res_2.t, rho2_flat.T, kind='linear', fill_value="extrapolate")
    rho2_interp_flat = interp_func(res.t).T            # (N, 36)
    rho2_interp = rho2_interp_flat.reshape((-1, 6, 6))  # (N, 6, 6)

    # Compute fidelity at each time point
    fidelities = [fidelity(r1, r2) for r1, r2 in zip(rho1_list, rho2_interp)]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(res.t, fidelities, label='Fidelity')
    plt.xlabel("Time")
    plt.ylabel("Fidelity")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.title("Fidelity between Density Matrices Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return fidelities