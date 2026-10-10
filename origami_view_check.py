"""Compare unchanged rigid-fold geometry using calibrated orthographic views."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import origami_grid3x3 as small
import origami_grid5x5 as large


BACKGROUND = '#f5f7fb'


def projected_points(axis, points):
    horizontal, vertical, _ = proj3d.proj_transform(*np.asarray(points).reshape(-1, 3).T, axis.get_proj())
    return axis.transData.transform(np.column_stack([horizontal, vertical]))


def pixels_per_unit(axis):
    """Two equal singular values certify an undistorted orthographic camera."""
    basis = projected_points(axis, np.vstack([np.zeros(3), np.eye(3)]))
    return np.linalg.svd((basis[1:] - basis[0]).T, compute_uv=False)


def render(size, output_directory):
    path = small.fold_path(80)
    if size == 3:
        configurations = np.array([small.folded_faces(angles)[0] for angles in path])
        vertices = np.array([small.grid_vertices(panels) for panels in configurations])
        colors = small.COLORS
        camera = (small.VIEW_ELEVATION, small.VIEW_AZIMUTH, 0)
    else:
        vertices, configurations = map(np.array, zip(*(large.grid_configuration(angles) for angles in path)))
        colors = large.COLORS
        camera = (large.VIEW_ELEVATION, large.VIEW_AZIMUTH, large.VIEW_ROLL)
    cameras = (camera, (90, -90, 0), (0, -90, 0))
    titles = ('Original oblique camera', 'Top view: X / Y', 'Side view: X / Z')
    figure = plt.figure(figsize=(14, 10), facecolor=BACKGROUND)
    layout = figure.add_gridspec(2, 3, left=0.05, right=0.98, top=0.85, bottom=0.20,
                                wspace=0.05, hspace=0.16)
    axes = []
    for row, state_index in enumerate((0, -1)):
        for column, (elevation, azimuth, roll) in enumerate(cameras):
            axis = figure.add_subplot(layout[row, column], projection='3d')
            axis.add_collection3d(Poly3DCollection(configurations[state_index], facecolors=colors,
                                                 edgecolors='#203349', linewidths=1.0))
            span = large.prepare_axis(axis, configurations)
            axis.view_init(elev=elevation, azim=azimuth, roll=roll)
            axis.set_title(titles[column] if row == 0 else 'Same camera / same scale',
                           fontsize=12, color='#46566a', pad=3)
            axes.append(axis)
    figure.canvas.draw()
    for axis in axes:
        large.fit_axis(axis, configurations, span)
    fitted_scales = np.array([pixels_per_unit(axis) for axis in axes])
    common_scale = fitted_scales.min()
    for axis in axes:
        axis.set_box_aspect(span, zoom=1)
        reset_scale = pixels_per_unit(axis)[0]
        axis.set_box_aspect(span, zoom=common_scale / reset_scale)
    scales = np.array([pixels_per_unit(axis) for axis in axes])
    np.testing.assert_allclose(scales, np.full_like(scales, common_scale), atol=1e-9, rtol=0)
    for axis in axes:
        pixels = projected_points(axis, configurations)
        if (pixels[:, 0].min() < axis.bbox.x0 or pixels[:, 0].max() > axis.bbox.x1
                or pixels[:, 1].min() < axis.bbox.y0 or pixels[:, 1].max() > axis.bbox.y1):
            raise AssertionError('A calibrated view clips the folding path.')
    row_column = np.array([small.row_column_spans(mesh) for mesh in vertices])
    changes = 100 * (row_column[-1] / row_column[0] - 1)
    image_ratios = []
    for column, title in enumerate(titles):
        initial = np.ptp(projected_points(axes[column], configurations[0]), axis=0)
        final = np.ptp(projected_points(axes[column + 3], configurations[-1]), axis=0)
        image_ratios.append(final / initial)
        print(f'{size}x{size} {title}: projected width/height changes = {100 * (final / initial - 1)}%')
    figure.suptitle(f'{size} × {size} View Check · Geometry Unchanged', fontsize=21, fontweight='bold', y=0.97)
    figure.text(0.5, 0.91, 'Orthographic views · One physical scale for all six panels · Full-edge hinges retained',
                ha='center', fontsize=12, color='#46566a')
    figure.text(0.02, 0.70, f'INITIAL\n{path[0, 0]:.2f}°', ha='center', va='center',
                fontsize=10, rotation=90, color='#46566a')
    figure.text(0.02, 0.36, f'FINAL\n{path[-1, 0]:.2f}°', ha='center', va='center',
                fontsize=10, rotation=90, color='#46566a')
    figure.text(0.5, 0.14, f'Spatial end-to-end distance: along a row {changes[0]:+.1f}%'
                f'  |  along a column {changes[1]:+.1f}%', ha='center', fontsize=12, color='#233e53')
    figure.text(0.5, 0.095, 'Grid directions are not screen axes. Projected silhouettes overlap; '
                'overlap in a view is not proof of a collision.', ha='center', fontsize=10, color='#46566a')
    figure.text(0.5, 0.05, 'No panel deformation, independent panel scaling, camera motion or per-frame zoom.',
                ha='center', fontsize=10, color='#46566a')
    output = output_directory / f'origami_grid{size}x{size}_view_check.png'
    figure.savefig(output, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)
    print(f'PASS: six matched, undistorted camera scales = {common_scale:.6f} px/unit; {output.resolve()}')
    return dict(spatial_changes=changes, image_ratios=np.array(image_ratios), camera_scales=scales)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=(3, 5))
    parser.add_argument('--output-directory', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    for size in (args.size,) if args.size else (3, 5):
        render(size, args.output_directory)


if __name__ == '__main__':
    main()
