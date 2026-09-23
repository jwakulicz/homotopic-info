import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
def plot_partial_trajectory(
    ax,
    trajectory,
    idx_time,
    color = 'k',
    alpha = 1
):
    past_lines = ax.plot(
        trajectory[:idx_time, 0],
        trajectory[:idx_time, 1],
        c = color,
        alpha = alpha,
        linestyle = 'solid'
    )

    future_lines = ax.plot(
        trajectory[idx_time:, 0],
        trajectory[idx_time:, 1],
        c = color,
        alpha = alpha,
        linestyle = '--'
    )

    return past_lines + future_lines

def plot_gmm(
    ax,
    weights,
    means,
    covariances,
    hsig_map,
    cmap = lambda idx: 'k',
    kwargs_mean = dict(),
    kwargs_cov = dict(),
    alpha_func = lambda weight: weight
):
    alphas = alpha_func(weights)
    alphas = alphas/alphas.max()
    alphas = np.clip(alphas, 0., 1.)
    for idx_components in range(weights.shape[0]):
        # Override user alpha
        kwargs_mean['alpha'] = 0.9 if alphas[idx_components] > 0.1 else alphas[idx_components]
        kwargs_mean['color'] = cmap(idx_components)
        kwargs_mean['label'] = hsig_map[idx_components]
        kwargs_cov['alpha'] = 0.5 * alphas[idx_components] if alphas[idx_components] > 0.1 else alphas[idx_components]
        kwargs_cov['facecolor'] = cmap(idx_components)

        plot_traj_with_covariance(
            ax,
            means[idx_components, ...],
            covariances[idx_components, ...],
            kwargs_mean,
            kwargs_cov
        )


def plot_traj_with_covariance(
    ax,
    mean,
    covariance,
    kwargs_mean,
    kwargs_dict
):
    # plot covariance
    for idx_time in range(mean.shape[0]):
        confidence_ellipse(
            ax,
            mean[idx_time, 0], 
            mean[idx_time, 1],
            covariance[idx_time, :, :],
            **kwargs_dict
        )

    ax.plot(
        mean[:, 0],
        mean[:, 1],
        **kwargs_mean
    )

    # ax.legend()

def confidence_ellipse(ax, mean_x, mean_y, cov, n_std=3.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    From https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y) .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
