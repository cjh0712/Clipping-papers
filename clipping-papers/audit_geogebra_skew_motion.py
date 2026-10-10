"""Independently check motion from the approved, unchanged K-branch assembly."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from build_geogebra_skew_grid5x5 import REFERENCE_ANGLE, original_points, reference_mesh


PANEL_DISTANCES = np.array([[0, 2, np.sqrt(8), 2], [2, 0, 2, 2],
                            [np.sqrt(8), 2, 0, 2], [2, 2, 2, 0]])


def outside_neighbor(origin, first, second, incoming):
    axis = first + second - 2 * origin
    incoming = incoming - origin
    return origin - incoming + 2 * np.dot(incoming, axis) / np.dot(axis, axis) * axis


def opposite_corner(origin, first, second, long_diagonal, sign):
    first, second = first - origin, second - origin
    middle = (2 / 3 if long_diagonal else 1 / 2) * (first + second)
    height = (np.sqrt(2) / 3 if long_diagonal else 1 / (2 * np.sqrt(2))) * np.cross(first, second)
    return origin + middle + sign * height


def eight_panel_patch(angle, signs=None):
    original = original_points(angle)
    _, reference, _, _ = reference_mesh()
    vertices = np.full((4, 4, 3), np.nan)
    for row, labels in enumerate((('O', 'D', 'C'), ('F', 'A', 'B'), ('K', 'G', 'Q'))):
        for column, label in enumerate(labels):
            vertices[row, column] = original[label]
    vertices[1, 3] = outside_neighbor(original['B'], original['C'], original['Q'], original['A'])
    vertices[3, 1] = outside_neighbor(original['G'], original['K'], original['Q'], original['A'])
    recipes = (
        ((0, 3), original['B'], original['C'], vertices[1, 3], False),
        ((2, 3), original['B'], vertices[1, 3], original['Q'], True),
        ((3, 0), original['G'], original['K'], vertices[3, 1], False),
        ((3, 2), original['G'], vertices[3, 1], original['Q'], True),
    )
    if signs is None:
        if not np.isclose(angle, REFERENCE_ANGLE, atol=1e-13, rtol=0):
            raise ValueError('Select branches at the approved initial pose, not at an animated pose.')
        signs = []
        for location, origin, first, second, long_diagonal in recipes:
            target = reference[location[0] + 1, location[1] + 1]
            signs.append(min((1, -1), key=lambda sign: np.linalg.norm(
                opposite_corner(origin, first, second, long_diagonal, sign) - target)))
    for sign, (location, origin, first, second, long_diagonal) in zip(signs, recipes):
        vertices[location] = opposite_corner(origin, first, second, long_diagonal, sign)
    return vertices, signs


def panel_error(vertices):
    errors = []
    for row in range(3):
        for column in range(3):
            if (row, column) == (2, 2):
                continue
            panel = vertices[[row, row, row + 1, row + 1],
                             [column, column + 1, column + 1, column]]
            distances = np.linalg.norm(panel[:, None] - panel[None, :], axis=-1)
            errors.append(np.max(np.abs(distances - PANEL_DISTANCES)))
    return float(max(errors))


def exact_certificate():
    coefficients, _, labels, _ = reference_mesh()
    points = coefficients[1:5, 1:5].reshape(-1, 3)
    labels = labels[1:5, 1:5].ravel()
    pairs = set()
    for row in range(3):
        for column in range(3):
            face = (4 * row + column, 4 * row + column + 1,
                    4 * (row + 1) + column + 1, 4 * (row + 1) + column)
            pairs.update(tuple(sorted(pair)) for pair in combinations(face, 2))
    pairs = sorted(pairs)
    fixed = {1, 2, 5, 6}
    free = [index for index in range(points.size) if index // 3 not in fixed]
    jacobian = sp.zeros(len(pairs), points.size)
    for row, (first, second) in enumerate(pairs):
        difference = (points[first] - points[second]) * [1, 1, 2]
        for axis in range(3):
            jacobian[row, 3 * first + axis] = int(difference[axis])
            jacobian[row, 3 * second + axis] = -int(difference[axis])
    reduced = jacobian[:, free]
    modes = reduced.nullspace()
    assert len(modes) == 1
    mode = modes[0] * (-2 / modes[0][free.index(12)])
    velocity = sp.zeros(points.size, 1)
    for index, value in zip(free, mode):
        velocity[index] = value
    assert list(velocity[12:15]) == [-2, 0, -1]
    right_side = sp.Matrix([
        -sum(weight * (velocity[3 * first + axis] - velocity[3 * second + axis]) ** 2
             for axis, weight in enumerate((1, 1, 2)))
        for first, second in pairs
    ])
    assert reduced * mode == sp.zeros(len(pairs), 1)
    certificate = next(stress for stress in reduced.T.nullspace()
                       if (stress.T * right_side)[0] != 0)
    assert certificate.T * reduced == sp.zeros(1, len(free))
    obstruction = (certificate.T * right_side)[0]
    assert obstruction == -32
    return {
        'patch': 'rows 2..4 and columns 2..4 of the approved 5x5',
        'coordinate_convention': 'physical point = (x, y, sqrt(2)*h)',
        'fixed_face': ['D', 'C', 'A', 'B'],
        'driven_velocity_in_scaled_coordinates': [-2, 0, -1],
        'free_coordinates': len(free),
        'distance_equations': len(pairs),
        'exact_rank': len(free) - len(modes),
        'infinitesimal_modes': len(modes),
        'certificate_times_jacobian': 'identically zero',
        'certificate_times_required_acceleration': str(obstruction),
        'consequence': 'The acceleration equations would require 0 = -32. The unique '
                       'nonzero infinitesimal mode cannot extend to a smooth rigid fold.',
        'certificate': [{'first': str(labels[first]), 'second': str(labels[second]),
                         'weight': str(certificate[index])}
                        for index, (first, second) in enumerate(pairs) if certificate[index]],
    }


def audit_motion():
    _, reference, _, _ = reference_mesh()
    initial, signs = eight_panel_patch(REFERENCE_ANGLE)
    mask = np.isfinite(initial).all(axis=-1)
    np.testing.assert_allclose(initial[mask], reference[1:5, 1:5][mask], atol=1e-12, rtol=0)
    samples = []
    maximum_error = 0.0
    for degrees in np.linspace(-10, 10, 401):
        vertices, _ = eight_panel_patch(REFERENCE_ANGLE + np.radians(degrees), signs)
        error = panel_error(vertices)
        maximum_error = max(maximum_error, error)
        if error > 1e-11:
            raise AssertionError(f'The eight-panel reference deformed: {error}')
        if any(np.isclose(degrees, value, atol=1e-10) for value in (-10, -5, -1, 0, 1, 5, 10)):
            length = float(np.linalg.norm(vertices[2, 3] - vertices[3, 2]))
            samples.append({'circle_angle_change_degrees': float(degrees),
                            'required_short_diagonal': 2.0,
                            'forced_short_diagonal': length,
                            'short_diagonal_mismatch': length - 2.0})
    assert abs(samples[3]['short_diagonal_mismatch']) < 1e-12
    assert all(sample['short_diagonal_mismatch'] > 0 for sample in samples if sample['circle_angle_change_degrees'])
    return {
        'status': 'blocked_by_rigid_closure_not_by_fixed_F',
        'initial_pose_unchanged': True,
        'K_replaced_by_J': False,
        'whole_edge_connections': 'shared vertex identities, not approximate contacts',
        'construction': 'Original four panels plus the two panels to their right and two below. '
                        'All eight retain their six distances and their initial intersection branches.',
        'closing_panel_in_5x5_one_based': [4, 4],
        'closing_panel_diagonal_endpoints': ['V_{34}', 'V_{43}'],
        'maximum_eight_panel_distance_error_over_401_samples': maximum_error,
        'samples': samples,
        'independent_exact_certificate': exact_certificate(),
    }


def diagnostic_preview(output):
    _, vertices, _, faces = reference_mesh()
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    figure = plt.figure(figsize=(12, 7), facecolor='#f7f9fc')
    axis = figure.add_axes([0.01, 0.12, 0.66, 0.76], projection='3d',
                           facecolor='#f7f9fc', computed_zorder=False)
    panels = vertices.reshape(-1, 3)[faces]
    triangles, colors, outlines = [], [], []
    for index, panel in enumerate(panels):
        row, column = divmod(index, 5)
        blue = (row + column) % 2 == 0
        corner_sets = ((0, 1, 3), (1, 2, 3)) if blue else ((0, 1, 2), (0, 2, 3))
        triangles.extend(panel[list(corners)] for corners in corner_sets)
        color = '#e9a2a7' if (row, column) == (3, 3) else '#c0d7e9' if blue else '#eddfac'
        colors.extend([color] * 2)
        outlines.append(panel[[0, 1, 2, 3, 0]])
    axis.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors=colors,
                                         linewidths=0, antialiaseds=False, zorder=1))
    axis.add_collection3d(Line3DCollection(outlines, colors='#687d8c', linewidths=0.8, zorder=2))
    source_perimeter = vertices[[1, 1, 1, 2, 3, 3, 3, 2, 1], [1, 2, 3, 3, 3, 2, 1, 1, 1]]
    axis.plot(*source_perimeter.T, color='#426b9c', linewidth=2, zorder=3)
    closing = panels[18]
    axis.plot(*closing[[0, 1, 2, 3, 0]].T, color='#ae3546', linewidth=2.8, zorder=4)
    endpoints = vertices[[3, 4], [4, 3]]
    axis.plot(*endpoints.T, color='#ae3546', linewidth=2, linestyle='--', zorder=5)
    for label, point in zip(('X', 'Y'), endpoints):
        axis.scatter(*point, color='#ae3546', s=25, depthshade=False, zorder=6)
        axis.text(*point, label, color='#8c2335', fontsize=14, zorder=7,
                  bbox={'fc': 'white', 'ec': 'none', 'alpha': 0.85, 'pad': 1})
    lower, upper = vertices.min(axis=(0, 1)), vertices.max(axis=(0, 1))
    axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
    axis.set_box_aspect(upper - lower)
    axis.set_proj_type('ortho')
    axis.view_init(elev=70, azim=25, roll=130)
    axis.set_axis_off()
    figure.suptitle('正确静态排列 ≠ 可保持刚性的联动', fontsize=20, fontweight='bold', y=0.95)
    figure.text(0.69, 0.79, '冲突位置：第4行、第4列', fontsize=14, color='#ae3546', weight='bold', va='top')
    figure.text(0.69, 0.72, 'X、Y 是该面的两个顶点。\n面片整体刚性要求 XY 始终为 2。',
                fontsize=11, color='#354c60', linespacing=1.8, va='top')
    figure.text(0.69, 0.56, 'F 沿原圆转动       联动所需 XY\n\n初态                    2.000000\n+1°                     2.001234\n−1°                     2.001204\n+5°                     2.032573',
                fontsize=11, color='#354c60', linespacing=1.7, va='top')
    figure.text(0.69, 0.24, '相邻8个面均保持刚性，\n但此处无法接上第9个刚性面。',
                fontsize=11, color='#ae3546', linespacing=1.8, va='top')
    figure.text(0.5, 0.055, '蓝框：原2×2，K与圆驱动均保留。红色虚线仅用于说明冲突，不是新增折痕。',
                ha='center', fontsize=10, color='#52677a')
    figure.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('origami_grid5x5_skew_motion_audit.json'))
    arguments = parser.parse_args()
    report = audit_motion()
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    diagnostic_preview(arguments.output.with_suffix('.png'))
    print(json.dumps({key: value for key, value in report.items()
                      if key != 'independent_exact_certificate'}, ensure_ascii=False, indent=2))
    print('Independent rational-arithmetic certificate: motion would require 0 = -32.')
    print('No foldable .ggb or GIF generated; the approved static model remains unchanged.')


if __name__ == '__main__':
    main()
