"""Miscellaneous utility functions for generating plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import IndetSnSpArray


def plot_indeterminate_analysis(
    df: pd.DataFrame, task: str = "depression", show: bool = True
):
    """Analyze performance achievable by including an indeterminate output option and plot.

    Parameters
    ----------
    df: Output of `kirad.utils.get_dataframe_with_scores`
    task: one of the tasks the model was trained on, for which scores and labels are in `df`
    show: Whether to show the plots; set to False for non-interactive workflows.

    """
    indet_sn_sp_arrays = dict()
    labels = df[f"quantized_labels_{task}"]
    scores = df[f"scores_{task}"]
    for i in range(labels.max()):
        indet_sn_sp_arrays[i] = IndetSnSpArray.build(
            y_true=(labels > i).astype(int), y_score=scores
        )

    for i, indet_sn_sp_array in indet_sn_sp_arrays.items():
        sn_eq_sp = indet_sn_sp_array.sn_eq_sp_graph()
        plt.figure(0)
        lines = plt.plot(
            sn_eq_sp.indet_frac, sn_eq_sp.lower_thresh, label=f"{task}>{i}"
        )
        plt.plot(sn_eq_sp.indet_frac, sn_eq_sp.upper_thresh, color=lines[0].get_color())
        plt.figure(1)
        plt.plot(
            sn_eq_sp.indet_frac,
            sn_eq_sp.min_sn_sp,
            label=f"{task}>{i}",
        )

    plt.figure(0)
    plt.grid(True)
    plt.legend()
    plt.xlabel("Indeterminate budget (fraction)")
    plt.ylabel("Thresholds to achieve optimal Sn=Sp given budget")
    plt.title("Upper and lower indeterminacy thresholds as a function of budget")

    plt.figure(1)
    plt.grid(True)
    plt.legend()
    plt.xlabel("Indeterminate budget (fraction)")
    plt.ylabel("Sn = Sp")
    plt.title("Sn=Sp as a function of indeterminate budget")

    for i, indet_sn_sp_array in indet_sn_sp_arrays.items():
        plt.figure()
        for budget in np.linspace(0.0, 0.7, 8):
            roc_curve = indet_sn_sp_array.roc_curve(budget)
            plt.plot(roc_curve.sp, roc_curve.sn, label=f"{budget=:.1f}")
        plt.xlim(1.0, 0.0)
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.xlabel("Specificity")
        plt.ylabel("Sensitivity")
        plt.title(f"ROC curves with indeterminacy budgets for {task}>{i}")
        plt.grid(True)
        plt.gca().set_aspect("equal")

    if show:
        plt.show()
