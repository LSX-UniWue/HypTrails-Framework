import numpy as np

def plot_hyptrails(evidences: dict, k_values: list, save_path: str) -> None:
    import matplotlib.pyplot as plt
    # plt.style.use('seaborn')
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 18,
        "font.size": 18,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
    width = 345
    plt.rcParams.update(tex_fonts)
    fig_size = set_size(width) 
    plt.figure(figsize=fig_size)
    for name, evidence in evidences.items():
        plt.plot(np.arange(len(evidence)), evidence, label=f"{name}")
    plt.xticks(np.arange(len(evidence)), k_values)
    plt.xlabel("k-factor")
    plt.ylabel("Evidence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 1.6

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim