"""Baseline run of the legacy 3D generator: per-stage timings and output statistics.

Runs grain sampling, RSA and tessellation of dragen.main3D exactly as the GUI would, with every
solver export switched off, and records

  * wall time of each stage (sampling incl. process_df, RSA, tessellation, total)
  * the label field straight out of the tessellation
  * an input-vs-output report from dragen.analysis.metrics

Example:
    venv/Scripts/python benchmarks/legacy_baseline.py --case lognormal --box 20 --resolution 2 --seed 1

Results go to benchmarks/results/legacy/<case>_box<box>_res<res>_seed<seed>/.
"""
import argparse
import functools
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from dragen.analysis import metrics  # noqa: E402

INPUT_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'inputs')
RESULT_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'legacy')
PHASES = ['Ferrite', 'Martensite', 'Pearlite', 'Bainite', 'Austenite', 'Inclusions', 'Bands']


def lognormal_input(seed: int = 0, n: int = 20000) -> str:
    """Write (once) a synthetic single-phase grain file: lognormal radii, elongated in x-y."""
    path = os.path.join(INPUT_DIR, 'lognormal_ferrite.csv')
    if not os.path.isfile(path):
        rng = np.random.default_rng(seed)
        a = rng.lognormal(mean=np.log(2.5), sigma=0.35, size=n)
        df = pd.DataFrame({'a': a, 'b': a * rng.uniform(0.6, 1.0, n), 'c': a,
                           'alpha': rng.uniform(0, 180, n)})
        os.makedirs(INPUT_DIR, exist_ok=True)
        df.to_csv(path, index=False)
    return path


CASES = {
    # single phase, csv input, no GAN involved
    'lognormal': lambda: ({1: lognormal_input()}, {1: 1.0}),
    # two phases from the trained GANs shipped in ExampleInput (same pair as tests/scenarios/Case_016)
    'gan_dual': lambda: ({2: os.path.join(REPO_ROOT, 'ExampleInput/Martensite/TrainedData_Martensite.pkl'),
                          5: os.path.join(REPO_ROOT, 'ExampleInput/Austenite/TrainedData_Austenite.pkl')},
                         {2: 0.5, 5: 0.5}),
}


class StageTimer:
    def __init__(self):
        self.seconds = {}
        self.captured = {}

    def wrap(self, owner, name, stage, capture=None):
        original = getattr(owner, name)

        @functools.wraps(original)
        def timed(*args, **kwargs):
            start = time.perf_counter()
            result = original(*args, **kwargs)
            self.seconds[stage] = self.seconds.get(stage, 0.0) + time.perf_counter() - start
            if capture:
                capture(self.captured, args, result)
            return result

        setattr(owner, name, timed)


def trace_tessellation_epochs(tessellation_cls) -> list:
    """Record active grains and filled voxels at the start of every tessellation epoch.

    Tesselation3D.run_tesselation is one monolithic loop; its only per-epoch hook is the
    np.random.shuffle of the active grain list, and the label array it grows is the `rsa`
    argument itself (modified in place). Both are intercepted while run_tesselation runs.
    """
    trace = []
    original_run = tessellation_cls.run_tesselation
    original_shuffle = np.random.shuffle

    def run_traced(self, rsa, *args, **kwargs):
        def shuffle(seq):
            trace.append({'active_grains': len(seq), 'filled': int(np.count_nonzero(rsa))})
            original_shuffle(seq)

        np.random.shuffle = shuffle
        try:
            rve, status = original_run(self, rsa, *args, **kwargs)
        finally:
            np.random.shuffle = original_shuffle
        trace.append({'active_grains': 0, 'filled': int(np.count_nonzero(rve)), 'total': int(rve.size)})
        return rve, status

    tessellation_cls.run_tesselation = run_traced
    return trace


def summarize_trace(trace: list) -> dict:
    """Split the tessellation into the regular growth phase and the reset (fill-up) phase."""
    if not trace:
        return {}
    total = trace[-1]['total']
    start = trace[0]['filled']
    reset_epoch = next((i for i in range(1, len(trace) - 1)
                        if trace[i]['active_grains'] > trace[i - 1]['active_grains']), None)
    filled_before_reset = trace[reset_epoch]['filled'] if reset_epoch is not None else trace[-1]['filled']
    return {
        'epochs': len(trace) - 1,
        'reset_epoch': reset_epoch,
        'fraction_after_rsa': round(start / total, 4),
        'fraction_from_regular_growth': round((filled_before_reset - start) / total, 4),
        'fraction_from_reset_phase': round((trace[-1]['filled'] - filled_before_reset) / total, 4),
        'filled_per_epoch': [round(t['filled'] / total, 4) for t in trace],
    }


def run(case: str, box: float, resolution: float, seed: int, out_dir: str = None) -> dict:
    """Generate one legacy RVE and return its timings, report, labels and grain table."""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    from dragen.run import Run
    from dragen.main3D import DataTask3D
    from dragen.generation.DiscreteRsa3D import DiscreteRsa3D
    from dragen.generation.DiscreteTesselation3D import Tesselation3D
    from dragen.utilities.Helpers import HelperFunctions
    from dragen.utilities.InputInfo import RveInfo

    file_entries, ratio_entries = CASES[case]()
    files = {i: file_entries.get(i) for i in range(1, 8)}
    phase_ratio = {i: ratio_entries.get(i, 0) for i in range(1, 8)}

    if out_dir is None:
        out_dir = os.path.join(RESULT_DIR, f'{case}_box{box:g}_res{resolution:g}_seed{seed}')
    os.makedirs(out_dir, exist_ok=True)

    Run(dimension=3, box_size=box, box_size_y=None, box_size_z=None, resolution=resolution,
        number_of_rves=1, slope_offset=0, abaqus_flag=False, damask_flag=False, moose_flag=False,
        calibration_rve_flag=False, element_type='HEX8', pbc_flag=False, submodel_flag=False,
        phase2iso_flag={i: True for i in range(1, 6)}, smoothing_flag=False, xfem_flag=False,
        gui_flag=False, anim_flag=False, root=out_dir, info_box_obj=None, progress_obj=None,
        phase_ratio=phase_ratio, file_dict=files, phases=PHASES, number_of_bands=0,
        upper_band_bound=0, lower_band_bound=0, band_orientation='xy', band_filling=0,
        subs_flag=False, subs_file_flag=False, subs_file=None, t_mu=1.0)
    Run.setup_logging()
    Run.initializations(0)

    timer = StageTimer()
    timer.wrap(HelperFunctions, 'process_df', 'process_df')
    timer.wrap(DiscreteRsa3D, 'run_rsa', 'rsa')
    timer.wrap(Tesselation3D, '__init__', 'tessellation_init',
               capture=lambda c, a, r: c.__setitem__('grains', a[1].copy()))
    timer.wrap(Tesselation3D, 'run_tesselation', 'tessellation',
               capture=lambda c, a, r: c.__setitem__('labels', r[0].copy()))
    trace = trace_tessellation_epochs(Tesselation3D)
    timer.wrap(DataTask3D, 'grain_sampling', 'sampling_total')
    timer.wrap(DataTask3D, 'rve_generation', 'generation_total')

    task = DataTask3D()
    total_df, _ = task.grain_sampling()
    task.rve_generation(total_df)

    labels = timer.captured['labels']
    grains = timer.captured['grains'].reset_index(drop=True)
    # tessellation labels grain row k with k + 1 (DiscreteTesselation3D.run_tesselation)
    target = np.concatenate([[0.0], grains['final_conti_volume'].to_numpy(float)])
    phase = np.concatenate([[0], grains['phaseID'].to_numpy(int)])
    rep = metrics.report(labels, target_volume=target, phase_of_grain=phase,
                         voxel_volume=RveInfo.bin_size ** 3, periodic=(True, True, True))

    result = {
        'case': case, 'box': box, 'resolution': resolution, 'seed': seed,
        'shape': list(labels.shape), 'n_grains': int(len(grains)),
        'seconds': {k: round(v, 3) for k, v in timer.seconds.items()},
        'report': rep.to_dict(),
        'target_phase_ratio': {k: v for k, v in phase_ratio.items() if v},
        'tessellation': summarize_trace(trace),
    }
    np.savez_compressed(os.path.join(out_dir, 'labels.npz'), labels=labels)
    grains.to_csv(os.path.join(out_dir, 'grains.csv'), index=False)
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    return {**result, 'labels': labels, 'grains': grains, 'report_obj': rep}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', choices=sorted(CASES), default='lognormal')
    parser.add_argument('--box', type=float, default=20.0, help='edge length in micrometres')
    parser.add_argument('--resolution', type=float, default=2.0, help='voxels per micrometre')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    result = run(args.case, args.box, args.resolution, args.seed)
    printable = {k: v for k, v in result.items() if k not in ('labels', 'grains', 'report_obj')}
    print(json.dumps(printable, indent=2))


if __name__ == '__main__':
    main()
