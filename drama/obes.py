import numpy as np
import pylcp
import scipy.stats as stats
from pathos.multiprocessing import Pool
from tqdm import tqdm
import os

def confidence_interval_68(data):
    n = len(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)  # Standard Error (SE)
    
    # 68% confidence interval for normal distribution
    z = stats.norm.ppf(0.84)  # 0.84 corresponds to one-sided 68%
    margin_of_error = z * std_err

    
    return margin_of_error

def return_two_level_hamiltonian(Delta, F_g=1, F_e=1, gF_g=1, gF_e=0):

    H_g, muq_g = pylcp.hamiltonians.singleF(F=F_g, gF=gF_g, muB=1) 
    H_e, muq_e = pylcp.hamiltonians.singleF(F=F_e, gF=gF_e, muB=1) 
    hamiltonian = pylcp.hamiltonian()

    hamiltonian.add_H_0_block('g', H_g)
    hamiltonian.add_H_0_block('e', -Delta*np.eye(2*F_e+1))

    hamiltonian.add_mu_q_block('g', muq_g, muB=1)
    hamiltonian.add_mu_q_block('e', muq_e, muB=1)

    hamiltonian.add_d_q_block('g','e', pylcp.hamiltonians.dqij_two_bare_hyperfine(F_g, F_e))

    return hamiltonian


def generate_force_curve(obe, pos, vel):
    import numpy as np
    return obe.generate_force_profile(pos, vel, deltat_v=2*np.pi*10, deltat_tmax=2*np.pi*200, itermax=10,
    rel=1e-10, abs=1e-12, method = 'DOP853')


def return_lasers(s, d, active_directions, phase, s_ratio=1, relative_phase=0.):
    """
    Construct a dictionary of laser beams for DRAMA simulation.

    Parameters:
    ----------
    s : float
        Saturation parameter of each laser beam.
    d : float
        Two-photon detuning for DRAMA. A positive value corresponds to a trapping force
        for a positive magnetic field gradient in the z direction.
    active_directions : dict
        Dictionary specifying which directions (axes and signs) have active beams.
        Format: {'x': ['+', '-'], 'y': ['+'], 'z': ['-'], ...}
    phase : list of float
        List of laser phases (in radians) for each beam.
        Must be of length directions of active beams (σ⁻ and σ⁺ are set to have same relative_phase).
    s_ratio : float, optional
        Ratio of saturation parameters for co propagating σ⁻ and σ⁺ beams, r = σ⁻/σ⁺. Default is 1
    relative_phase : float, optional
        Relative phase between σ⁻ and σ⁺ beams. Default is 0.

    Returns:
    -------
    dict
        A dictionary containing one entry: 'g->e', which maps to a pylcp.laserBeams object.
    """
    laserBeams = {}
    beam_list = []
    i = 0  # Index for phase array

    # Unit vectors for beam propagation
    kvecs = {
        'x': np.array([1., 0., 0.]),
        'y': np.array([0., 1., 0.]),
        'z': np.array([0., 0., 1.]),
    }

    # Loop through axes and directions
    for axis in ['x', 'y', 'z']:
        if axis in active_directions:
            for direction in active_directions[axis]:
                sign = 1.0 if direction == '+' else -1.0
                kvec = sign * kvecs[axis]

                # Determine polarizations for σ- and σ+ depending on axis
                if axis == 'z':
                    pol_sigma_minus = +1
                    pol_sigma_plus = -1
                else:
                    pol_sigma_minus = -1
                    pol_sigma_plus = +1

                # σ- beam
                beam_list.append({
                    'kvec': kvec,
                    'pol': pol_sigma_minus,
                    'delta': d/2,
                    's': s*s_ratio,
                    'phase': phase[i]
                })

                # σ+ beam
                beam_list.append({
                    'kvec': kvec,
                    'pol': pol_sigma_plus,
                    'delta': -d/2,
                    's': s,
                    'phase': phase[i] + relative_phase
                })
                i += 1

    laserBeams['g->e'] = pylcp.laserBeams(beam_list, beam_type=pylcp.infinitePlaneWaveBeam)
    return laserBeams


def get_obe(
    B: float,
    v: float,
    s: float,
    D: float,
    d: float,
    F_g=1, F_e=1, gF_g=1, gF_e=0,
    active_directions: dict = {'x': ['+', '-'], 'y': ['+', '-'], 'z': ['+', '-']},
    phase: np.ndarray = np.random.uniform(0, 2 * np.pi, 12),
    relative_phase: float = 0.0,
    s_ratio: float = 1.0,
    random_pos: bool = True,
    Bx: float = 0.0,
    By: float = 0.0,
):
    """
    Constructs and returns a pylcp.obe object for DRAMA dynamics.

    Parameters
    ----------
    B : float
        Magnetic field strength along the z-axis.
    v : float
        Velocity of the atom along the z-axis.
    s : float
        Saturation parameter for the laser beams.
    D : float
        Global detuning (used to build the Hamiltonian).
    d : float
        Two-photon detuning for DRAMA interaction.
    F_g : int
        F quantum number of the ground state.
    F_e : int
        F quantum number of the excited state.
    gF_g : float
        g factor of the ground state.
    gF_e : float
        g factor of the excited state.
    active_directions : dict, optional
        Dictionary specifying active laser beam directions. Each axis ('x', 'y', 'z') maps to a list of
        directions ['+', '-'] indicating the beam propagation. Default is full 3D setup.
    phase : np.ndarray, optional
        List of laser beam phases (in radians). Should have a length of number of active beams.
        Default is 6 random phases in [0, 2π].
    relative_phase : float, optional
        Relative phase between σ⁻ and σ⁺ beams. Default is 0.
    s_ratio : float, optional
        Ratio of saturation parameters for σ⁻ over σ⁺ beams. Default is 1
    random_pos : bool, optional
        Whether to initialize the atom at a random 3D position drawn from a normal distribution.
        If False, position is set to the origin. Default is True.
    Bx : float, optional
        Magnetic field along the x-axis. Default is 0.0.
    By : float, optional
        Magnetic field along the y-axis. Default is 0.0.

    Returns
    -------
    pylcp.obe
        An initialized OBE object ready for time evolution or force calculation.
    """

    # Construct the atomic Hamiltonian
    ham = return_two_level_hamiltonian(D, F_g, F_e, gF_g, gF_e)

    # Generate laser beam configuration based on active directions and phases
    lasers = return_lasers(s=s, d=d, active_directions=active_directions, phase=phase, s_ratio=s_ratio, relative_phase=relative_phase)

    # Create a constant magnetic field along z
    magField = pylcp.constantMagneticField([Bx, By, B])

    # Initialize the OBE object
    obe = pylcp.obe(lasers, magField, ham, transform_into_re_im=True)

    # Set initial position and velocity
    if random_pos:
        position = np.random.normal(0, 2 * np.pi, 3)
    else:
        position = [0., 0., 0.]

    velocity = [0., 0., v]
    obe.set_initial_position_and_velocity(position, velocity)

    return obe

from collections import defaultdict

def _detect_cpu_quota(default_fallback: int = 1) -> int:
    """
    Detect a safe number of worker processes under common schedulers.
    Takes the minimum of all hints (PBS, Slurm, CPU affinity, etc.).
    """
    hints = []

    # PBS / Torque / generic environment hints
    # for var in ("PBS_NP", "NCPUS", "OMP_NUM_THREADS"):
    #     v = os.environ.get(var)
    #     if v and v.isdigit():
    #         hints.append(int(v))
    #         print(f"PBS cpu quota hint: {int(v)}")

    # Slurm hints
    # for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
    #     v = os.environ.get(var)
    #     if v and v.isdigit():
    #         hints.append(int(v))
    #         print(f"Slurm cpu quota hint: {int(v)}")

    # Linux CPU affinity
    try:
        hints.append(len(os.sched_getaffinity(0)))
        print(f"Linux cpu quota hint: {len(os.sched_getaffinity(0))}")
    except Exception:
        pass

    # System CPU count (upper bound)
    try:
        hints.append(os.cpu_count() or default_fallback)
        print(f"Sustem cpu quota hint: {os.cpu_count() or default_fallback}")
    except Exception:
        pass

    # Pick the smallest positive hint (most conservative)
    valid = [h for h in hints if isinstance(h, int) and h > 0]
    if valid:
        n = max(valid)
    else:
        n = default_fallback

    return max(1, n)



def _set_single_thread_env():
    """
    Force all common threaded math libs to 1 thread inside each worker.
    This is crucial to avoid oversubscribing cluster CPU quotas.
    """
    env_vars = [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"
    ]
    for v in env_vars:
        os.environ[v] = "1"

def _pool_initializer():
    # Called once in each worker
    _set_single_thread_env()

def compute_forces_bv_to_files_parallel(
    bv_list: list,
    Delta: float,
    delta: float,
    s: float,
    it_num: int,
    s_ratio: float = 1.0,
    Bx: float = 0.0,
    By: float = 0.0,
    F_g=1, F_e=1, gF_g=1, gF_e=0,
    active_directions={'x': ['+', '-'], 'y': ['+', '-'], 'z': ['+', '-']},
    output_dir: str = "results/force_bv",
    print_progress: bool = False,
    random_relative_phase: bool = False,
    n_workers = None
):
    """
    Compute forces for a list of (B, v) pairs using a flat parallel task list and save each result to .npz files.
    Use this to scan B and v
    """

    _set_single_thread_env()
    if n_workers is None:
        n_workers = _detect_cpu_quota()
    os.makedirs(output_dir, exist_ok=True)
    print("Using worker number: ", n_workers)
    task_list = []
    bv_metadata = {}

    # Construct the full list of (B, v, phase) tasks
    for B, v in bv_list:
        filename = f"bv_b{B:+.3f}_v{v:+.3f}_d{delta:+.3f}_D{Delta:+.3f}_s{s:+.3f}_sratio{s_ratio:+.3f}_Bx{Bx:+.3f}".replace('.', 'p')
        filepath = os.path.join(output_dir, filename + ".npz")

        if os.path.exists(filepath):
            if print_progress:
                print(f"[SKIP] Already exists: {filepath}")
            continue

        bv_metadata[(B, v)] = filepath

        for _ in range(it_num):
            phase = np.random.uniform(0, 2 * np.pi, 7)
            if not random_relative_phase:
                phase[6] = 0.0  # Set relative phase to zero if not random
            task_list.append((B, v, phase))

    def run_single_task(B, v, phase):

        obe = get_obe(
            B=B,
            v=v,
            s=s,
            D=Delta,
            d=delta,
            active_directions=active_directions,
            phase=phase,
            relative_phase=phase[6],  # Use the last phase as the relative phase
            s_ratio=s_ratio,
            Bx=Bx,
            By=By,
            F_g=F_g,
            F_e=F_e,
            gF_g=gF_g,
            gF_e=gF_e,
        )

        position = np.random.normal(0, 2 * np.pi, 3)
        velocity = [0., 0., v]

        result = generate_force_curve(obe, position, velocity)
        return (B, v, result.F[2], phase)

    # Run all simulations in parallel
    from functools import partial

    with Pool(processes=n_workers,
              initializer=_pool_initializer) as pool:
        results_iter = pool.imap_unordered(lambda args: run_single_task(*args), task_list)
        all_results = []
        for result in tqdm(results_iter, total=len(task_list), desc="Running OBE tasks"):
            all_results.append(result)


    # Group results by (B, v)
    results_by_bv = defaultdict(list)
    phases_by_bv = defaultdict(list)

    for B, v, fz, phase in all_results:
        results_by_bv[(B, v)].append(fz)
        phases_by_bv[(B, v)].append(phase)

    # Save each file with grouped results
    for (B, v), fz_values in results_by_bv.items():
        fz_values = np.array(fz_values)
        phase_array = np.array(phases_by_bv[(B, v)])
        filepath = bv_metadata[(B, v)]

        mean_force = np.mean(fz_values)
        err = confidence_interval_68(fz_values)

        np.savez(
            filepath,
            b=B,
            v=v,
            Delta=Delta,
            delta=delta,
            s=s,
            s_ratio=s_ratio,
            phases=phase_array,
            forces=fz_values,
            mean_force=mean_force,
            err=err,
            Bx=Bx,
            By=By
        )

        if print_progress:
            print(f"[OK] B={B}, v={v}, mean_force={mean_force:.4f}, err={err:.2e}")


def compute_forces_scan_parallel(
    bv_list: list,
    Delta_list: list,
    delta_list: list,
    s_list: list,
    it_num: int,
    s_ratio: float = 1.0,
    Bx: float = 0.0,
    By: float = 0.0,
    F_g=1, F_e=1, gF_g=1, gF_e=0,
    active_directions={'x': ['+', '-'], 'y': ['+', '-'], 'z': ['+', '-']},
    output_dir: str = "results/force_bv",
    print_progress: bool = False,
    random_relative_phase: bool = False,
):
    """
    Compute forces for a scan over (B, v, Delta, delta, s), run all combinations in parallel, save grouped results.
    Use this to scan Delta, delta and s
    """

    _set_single_thread_env()

    os.makedirs(output_dir, exist_ok=True)

    n_workers = _detect_cpu_quota()
    n_workers = max(1, int(n_workers))
    print("Detected worker number: ", n_workers)

    # -------- Build full scan list -------- #
    scan_param_list = []
    file_registry = {}

    for B, v in bv_list:
        for Delta in Delta_list:
            for delta in delta_list:
                for s in s_list:
                    filename = f"bv_b{B:+.3f}_v{v:+.3f}_d{delta:+.3f}_D{Delta:+.3f}_s{s:+.3f}_sratio{s_ratio:+.3f}_Bx{Bx:+.3f}".replace('.', 'p')
                    filepath = os.path.join(output_dir, filename + ".npz")

                    if os.path.exists(filepath):
                        if print_progress:
                            print(f"[SKIP] Already exists: {filepath}")
                        continue

                    file_registry[(B, v, Delta, delta, s)] = filepath

                    for _ in range(it_num):
                        phase = np.random.uniform(0, 2 * np.pi, 7)
                        if not random_relative_phase:
                            phase[6] = 0.0  # fix relative phase
                        scan_param_list.append((B, v, Delta, delta, s, phase))

    # -------- Simulation function -------- #
    def run_single_task(B, v, Delta, delta, s, phase):
        np.random.seed()
        obe = get_obe(
            B=B,
            v=v,
            s=s,
            D=Delta,
            d=delta,
            active_directions=active_directions,
            phase=phase,
            relative_phase=phase[6],
            s_ratio=s_ratio,
            Bx=Bx,
            By=By,
            F_g=F_g,
            F_e=F_e,
            gF_g=gF_g,
            gF_e=gF_e,
        )
        position = np.random.normal(0, 2 * np.pi, 3)
        velocity = [0., 0., v]
        result = generate_force_curve(obe, position, velocity)

        return (B, v, Delta, delta, s, result.F[2], phase)

    # -------- Run in parallel with progress bar -------- #
    all_results = []
    with Pool(processes=n_workers,
              initializer=_pool_initializer) as pool:
        results_iter = pool.imap_unordered(lambda args: run_single_task(*args), scan_param_list)
        for result in tqdm(results_iter, total=len(scan_param_list), desc="Running OBE tasks"):
            all_results.append(result)

    # -------- Group and Save Results -------- #
    grouped_results = defaultdict(list)
    grouped_phases = defaultdict(list)

    for B, v, Delta, delta, s, fz, phase in all_results:
        key = (B, v, Delta, delta, s)
        grouped_results[key].append(fz)
        grouped_phases[key].append(phase)

    for key in grouped_results:
        B, v, Delta, delta, s = key
        fz_values = np.array(grouped_results[key])
        phase_array = np.array(grouped_phases[key])
        filepath = file_registry[key]

        mean_force = np.mean(fz_values)
        err = confidence_interval_68(fz_values)

        np.savez(
            filepath,
            b=B,
            v=v,
            Delta=Delta,
            delta=delta,
            s=s,
            s_ratio=s_ratio,
            phases=phase_array,
            forces=fz_values,
            mean_force=mean_force,
            err=err,
            Bx=Bx,
            By=By
        )

        if print_progress:
            print(f"[OK] B={B}, v={v}, D={Delta}, d={delta}, s={s}, ⟨Fz⟩={mean_force:.4f}, err={err:.2e}")


def compute_forces_var_B(
    b_list,
    Delta: float,
    delta: float,
    s: float,
    it_num: int,
    s_ratio: float = 1.0,
    F_g=1, F_e=1, gF_g=1, gF_e=0,
    active_directions={'x': ['+', '-'], 'y': ['+', '-'], 'z': ['+', '-']},
    output_dir: str = "results/force_bv",
    print_progress: bool = False
):
    """
    Compute forces for a list of B and save each result to a separate .npz file.

    Parameters
    ----------
    b_list : list of list
        A list of [Bx, By, Bz]
    Delta : float
        Global detuning.
    delta : float
        Two-photon detuning.
    s : float
        Saturation parameter.
    s_ratio : float, optional
        Ratio of saturation parameters for σ⁻ over σ⁺ beams. Default is 1
    F_g : int
        F quantum number of the ground state.
    F_e : int
        F quantum number of the excited state.
    gF_g : float
        g factor of the ground state.
    gF_e : float
        g factor of the excited state.
    it_num : int
        Number of Monte Carlo samples per magnetic field.
    active_directions : dict
        Active laser beam directions.
    output_dir : str
        Directory to save the output .npz files.
    """

    os.makedirs(output_dir, exist_ok=True)

    for B in tqdm(b_list, desc="Processing", unit="point"):
    # Check if file exists
        filename = f"scan_bx{B[0]:+.3f}_by{B[1]:+.3f}_bz{B[2]:+.3f}_d{delta:+.3f}_D{Delta:+.3f}_s{s:+.3f}_sratio{s_ratio:+.3f}".replace('.', 'p')
        filepath = os.path.join(output_dir, filename + ".npz")
        if os.path.exists(filepath):
            if print_progress:
                print(f"[SKIP] Already exists: {filepath}")
            continue

        arg_list = []
        phase_list = []

        for _ in range(it_num):
            phase = np.random.uniform(0, 2 * np.pi, 6)
            phase_list.append(phase)

            obe = get_obe(
                B=B[2],
                v=0.,
                s=s,
                D=Delta,
                d=delta,
                active_directions=active_directions,
                phase=phase,
                s_ratio=s_ratio,
                Bx=B[0],
                By=B[1],
                F_g=F_g,
                F_e=F_e,
                gF_g=gF_g,
                gF_e=gF_e,)

            position = np.random.normal(0, 2 * np.pi, 3)
            velocity = [0., 0., 0.]
            arg_list.append((obe, position, velocity))

        with Pool() as pool:
            result = pool.starmap(generate_force_curve, arg_list)

        unit_vector = np.array([-B[0], -B[1], B[2]/2]) / np.linalg.norm([-B[0], -B[1], B[2]/2])
        f_values = np.array([np.dot(res.F, unit_vector) for res in result])
        mean_force = np.mean(f_values)
        err = confidence_interval_68(f_values)
        phase_array = np.array(phase_list)
        theta = np.arctan2(B[1], B[0]) if (B[0] != 0 or B[1] != 0) else 0.0
        phi = np.arctan2(np.sqrt(B[0]**2 + B[1]**2), B[2]) if B[2] != 0 else np.pi/2

        np.savez(
            filepath,
            B=B,
            Delta=Delta,
            delta=delta,
            s=s,
            s_ratio=s_ratio,
            phases=phase_array,
            forces=f_values,
            mean_force=mean_force,
            err=err,
            theta=theta,
            phi=phi
        )
        if print_progress:
            print(f"Finish calculation: B={B}, D={Delta}, d={delta}, s={s}, sratio={s_ratio}, mean_force={mean_force}, err={err}")


from scipy.optimize import curve_fit
from itertools import product

def evolve_dens_to_files_parallel(
    bv_list,
    Delta,
    delta,
    s,
    time: float = 1e3,
    it_num: int = 8,
    s_ratio: float = 1.0,
    Bx: float = 0.0,
    By: float = 0.0,
    F_g=1, F_e=1, gF_g=1, gF_e=0,
    active_directions={'x': ['+', '-'], 'y': ['+', '-'], 'z': ['+', '-']},
    output_dir: str = "results/pop_bv",
    print_progress: bool = False
):
    """
    Run OBE evolutions and save per-(B, v, Delta, delta, s) result files.
    Delta, delta, s can be scalars or iterables.
    """

    # Normalize to lists
    def _as_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        return [x]

    Delta_list = _as_list(Delta)
    delta_list = _as_list(delta)
    s_list     = _as_list(s)

    os.makedirs(output_dir, exist_ok=True)

    # Create flat task list over all combinations
    # Each task: (B, v, D, d, s_val, phase)
    task_list = []
    filekey_to_path = {}

    for (B, v), D, d, s_val in product(bv_list, Delta_list, delta_list, s_list):
        # unique, filesystem-safe filename
        filename = (
            f"pop_b{B:+.3f}_v{v:+.3f}_d{d:+.3f}_D{D:+.3f}_s{s_val:+.3f}"
            f"_sratio{s_ratio:+.3f}_Bx{Bx:+.3f}_By{By:+.3f}"
        ).replace('.', 'p')
        filepath = os.path.join(output_dir, filename + ".npz")

        # Skip if file already exists
        if os.path.exists(filepath):
            if print_progress:
                print(f"[SKIP] Already exists: {filepath}")
            continue

        key = (B, v, D, d, s_val)
        filekey_to_path[key] = filepath

        # Enqueue it_num random phases for this parameter set
        for _ in range(it_num):
            task_list.append((B, v, D, d, s_val, np.random.uniform(0, 2 * np.pi, 6)))

    if not task_list:
        if print_progress:
            print("[INFO] No tasks to run (all outputs present?).")
        return

    def extract_cycle_average(t, y):
        def sine_func(t, A, f, phi, C):
            return A * np.sin(2 * np.pi * f * t + phi) + C

        # Use the last slice to fit a frequency, then average the last ~3 cycles
        t_fit = t[-10000:] if len(t) >= 10000 else t
        y_fit = y[-10000:] if len(y) >= 10000 else y

        A_guess = (np.max(y_fit) - np.min(y_fit)) / 2 if len(y_fit) > 0 else 0.0
        f_guess = 10 / (t_fit[-1] - t_fit[0]) if t_fit[-1] > t_fit[0] else 1.0
        p0 = [A_guess, f_guess, 0.0, float(np.mean(y_fit)) if len(y_fit) > 0 else 0.0]

        try:
            params, _ = curve_fit(sine_func, t_fit, y_fit, p0=p0, maxfev=10000)
            f = params[1]
            # guard against nonsense fits
            if not np.isfinite(f) or f <= 0:
                f = f_guess
        except Exception:
            f = f_guess

        # Average last ~3 cycles (clip to [0, time])
        if f <= 0:
            return float(np.mean(y[-1000:])) if len(y) >= 1000 else float(np.mean(y))
        t_lo = max(0.0, time - 3.0 / f)
        idx = np.where((t >= t_lo) & (t <= time))[0]
        if idx.size == 0:
            return float(np.mean(y[-1000:])) if len(y) >= 1000 else float(np.mean(y))
        return float(y[idx].mean())

    # Per-sample worker
    def _run_single_sample(args):
        B, v, D, d, s_val, phase = args

        obe = get_obe(
            B=B,
            v=v,
            s=s_val,
            D=D,
            d=d,
            active_directions=active_directions,
            phase=phase,
            s_ratio=s_ratio,
            Bx=Bx,
            By=By,
            F_g=F_g,
            F_e=F_e,
            gF_g=gF_g,
            gF_e=gF_e,
        )
        # Example initial conditions (same as original)
        obe.set_initial_rho_from_populations(np.array([1., 1., 1., 0., 0., 0.]) / 6)
        obe.set_initial_position_and_velocity(np.random.normal(0, 2 * np.pi, 3), [0., 0., v])

        res = obe.evolve_density([0, time], max_step=1e-1, rtol=1e-12, atol=1e-14, method="DOP853")

        rho00 = np.real(res.rho[0, 0])
        rho11 = np.real(res.rho[1, 1])
        rho22 = np.real(res.rho[2, 2])
        rho_exc = np.real(res.rho[3, 3] + res.rho[4, 4] + res.rho[5, 5])

        return (
            B, v, D, d, s_val,
            extract_cycle_average(res.t, rho00),
            extract_cycle_average(res.t, rho11),
            extract_cycle_average(res.t, rho22),
            extract_cycle_average(res.t, rho_exc),
            phase
        )

    # Parallel execution
    all_results = []
    with Pool() as pool:
        for result in tqdm(pool.imap_unordered(_run_single_sample, task_list),
                           total=len(task_list), desc="Running OBE samples"):
            all_results.append(result)

    # Group by parameter key
    results_by_key = defaultdict(list)
    phases_by_key = defaultdict(list)

    for B, v, D, d, s_val, rho00, rho11, rho22, exc, phase in all_results:
        key = (B, v, D, d, s_val)
        results_by_key[key].append((rho00, rho11, rho22, exc))
        phases_by_key[key].append(phase)

    # Save one file per (B, v, D, d, s)
    for key, pops in results_by_key.items():
        B, v, D, d, s_val = key
        filepath = filekey_to_path.get(key)
        if filepath is None:
            # (Shouldn't happen; guard in case of race/skip)
            continue

        pops = np.array(pops, dtype=float)
        rho00_arr = pops[:, 0]
        rho11_arr = pops[:, 1]
        rho22_arr = pops[:, 2]
        exc_arr   = pops[:, 3]
        phase_arr = np.array(phases_by_key[key])

        np.savez(
            filepath,
            b=B,
            v=v,
            Delta=D,
            delta=d,
            s=s_val,
            s_ratio=s_ratio,
            phases=phase_arr,
            rho00=rho00_arr,
            rho11=rho11_arr,
            rho22=rho22_arr,
            excited=exc_arr,
            mean_rho00=float(np.mean(rho00_arr)),
            mean_rho11=float(np.mean(rho11_arr)),
            mean_rho22=float(np.mean(rho22_arr)),
            mean_exc=float(np.mean(exc_arr)),
            err_rho00=confidence_interval_68(rho00_arr),
            err_rho11=confidence_interval_68(rho11_arr),
            err_rho22=confidence_interval_68(rho22_arr),
            err_excited=confidence_interval_68(exc_arr),
            Bx=Bx,
            By=By
        )

        if print_progress:
            print(f"[OK] B={B:+.3f}, v={v:+.3f}, D={D:+.3f}, d={d:+.3f}, s={s_val:+.3f} "
                  f"⟨ρₑ⟩={np.mean(exc_arr):.4f} ± {confidence_interval_68(exc_arr):.2e}")


    
import os
import numpy as np

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


def load_forces_var_B(output_dir: str = "results/force_bv", include_samples: bool = False):
    """
    Load results produced by `compute_forces_var_B`.

    Parameters
    ----------
    output_dir : str
        Directory containing the saved .npz files.
    include_samples : bool
        If True, include raw sample arrays for phases and forces.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'B': array of shape (N, 3) with [Bx, By, Bz] vectors
        - 'mean_force': array of mean projected forces
        - 'err': array of errors (68% CI)
        - 'delta', 'Delta', 's', 's_ratio': metadata values (assumes consistent)
        - 'theta', 'phi': angles of B field
        - Optional keys if include_samples is True:
            - 'phases_list': list of phase arrays
            - 'forces_list': list of force arrays
    """
    Bs = []
    mean_forces = []
    errs = []
    phases_list = []
    forces_list = []
    theta_list = []
    phi_list = []

    metadata = {
        'delta': None,
        'Delta': None,
        's': None,
        's_ratio': None
    }

    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".npz") or not fname.startswith("scan_bx"):
            continue

        data = np.load(os.path.join(output_dir, fname), allow_pickle=True)

        Bs.append(data['B'])
        mean_forces.append(data['mean_force'].item())
        errs.append(data['err'].item())
        theta_list.append(data['theta'].item())
        phi_list.append(data['phi'].item())

        if include_samples:
            forces_list.append(data['forces'])
            phases_list.append(data['phases'])

        # Store metadata from first file
        if metadata['delta'] is None:
            metadata['delta'] = data['delta'].item()
            metadata['Delta'] = data['Delta'].item()
            metadata['s'] = data['s'].item()
            metadata['s_ratio'] = data['s_ratio'].item()

    results = {
        'B': np.array(Bs),  # shape (N, 3)
        'mean_force': np.array(mean_forces),
        'err': np.array(errs),
        'theta': np.array(theta_list),
        'phi': np.array(phi_list),
        **metadata
    }

    if include_samples:
        results['phases_list'] = phases_list
        results['forces_list'] = forces_list

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
        "d": np.array(ds),
        "D": np.array(Ds),
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


def load_data(x_par, var_par, output_dir):
    """
    Load force simulation data from .npz files in a given directory, extracting specified parameters and results.

    Parameters
    ----------
    x_par : str
        The key name to extract for the x-axis variable (e.g., "B", "sratio", "Bx").
    var_par : str
        The key name to extract for grouping or labeling (e.g., "Delta", "s", "Bx").
    output_dir : str
        Path to the directory containing `.npz` result files. Each file must contain at least:
        - x_par (scalar)
        - 'mean_force' (scalar)
        - 'err' (scalar)
        - var_par (scalar)

    Returns
    -------
    x_list : np.ndarray
        Sorted array of values corresponding to `x_par`.
    y_list : np.ndarray
        Sorted array of `mean_force` values associated with each `x_par`.
    err_list : np.ndarray
        Sorted array of error estimates associated with each `mean_force`.
    var_list : np.ndarray
        Sorted array of values for `var_par`, matched to entries in `x_list`.

    Notes
    -----
    - Files must be `.npz` format and start with 'bv' in their filename.
    - If any file is missing a required key, it will be skipped with a warning.
    - The output arrays are sorted by `x_list`.

    Examples
    --------
    >>> x, f, e, delta = load_data(x_par="B", var_par="Delta", output_dir="results/force_bv")
    >>> plt.errorbar(x, f, yerr=e)

    """

    x_list = []
    y_list = []
    err_list = []
    var_list = []

    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith('.npz'):
            continue
        if not fname.startswith('bv'):
            continue
        path = os.path.join(output_dir, fname)
        data = np.load(path)

        try:
            x = data[x_par].item()
            y = data['mean_force'].item()
            e = data['err'].item()
            var = data[var_par].item()
        except KeyError as e:
            print(f"[WARNING] Skipping file {fname}, missing key: {e}")
            continue

        x_list.append(x)
        y_list.append(y)
        err_list.append(e)
        var_list.append(var)

    # Sort by sratio
    sort_idx = np.argsort(x_list)
    x_list = np.array(x_list)[sort_idx]
    y_list = np.array(y_list)[sort_idx]
    err_list = np.array(err_list)[sort_idx]
    var_list = np.array(var_list)[sort_idx]


    return x_list, y_list, err_list, var_list
