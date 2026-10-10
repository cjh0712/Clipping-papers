"""Assemble and audit nine congruent rigid nonplanar quadrilaterals.

The assembly follows the middle 3x3 patch of GeoGebra mj826xqp.
Each quadrilateral is a rigid pair of isosceles triangles, joined along
their short diagonal. The reference assembly is second-order locked when
these internal dihedrals are fixed; do not animate a nonphysical flex.
"""

import argparse
import json
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import least_squares


SOURCE_URL = 'https://www.geogebra.org/calculator/mj826xqp'
OUTPUT = Path(__file__).with_name('origami_spatial_quads3x3_assembled.png')
POINT_LABELS = ('F', 'G', 'B1', 'C1', 'E', 'B', 'C', 'D1',
                'B2', 'A', 'D', 'I', 'A1', 'D2', 'H', 'J')
VIEW_ELEVATION = 40
VIEW_AZIMUTH = -62
GRID_ELEVATION = 30
GRID_AZIMUTH = 0
GRID_ROLL = 45


@dataclass
class Geometry:
    positions: np.ndarray
    cells: np.ndarray
    triangles: np.ndarray
    pairs: np.ndarray
    squared_lengths: np.ndarray
    free: np.ndarray
    columns: np.ndarray
    driven: np.ndarray
    hinge: np.ndarray
    leg: float
    base: float


def rotation(axis, angle):
    """Right-handed rigid rotation, with angle in radians."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    skew = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
    return (np.eye(3) * np.cos(angle)
            + (1 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * skew)


def make_geometry(leg=2.45, base=2.0):
    """Crop the GeoGebra grid; retain its long diagonal and height pattern.

    Source coordinates use leg=sqrt(6), base=2 and long diagonal=4.
    Keeping the long diagonal at twice the base, adjust only the height
    to honor an exact leg length of 2.45 instead of its rounded display.
    """
    if not np.isfinite([leg, base]).all() or base <= 0 or leg <= np.sqrt(1.25) * base:
        raise ValueError('Need base > 0 and leg > sqrt(1.25)*base for a nonplanar unit.')
    height = np.sqrt(leg ** 2 - 1.25 * base ** 2)
    positions = np.array([
        [base * (row + column - 3), base * (column - row) / 2,
         height * (((row + column + 1) % 2) - 0.5)]
        for row in range(4) for column in range(4)
    ])
    cells = np.array([
        [4 * row + column, 4 * (row + 1) + column,
         4 * (row + 1) + column + 1, 4 * row + column + 1]
        for row in range(3) for column in range(3)
    ])
    triangles = np.array([triangle for cell in cells
                          for triangle in (cell[[0, 1, 3]], cell[[1, 2, 3]])])
    driven = cells[1]
    hinge = np.array([5, 6])
    fixed = set(cells[4]) | set(driven)
    free = np.array([index for index in range(16) if index not in fixed])
    pairs = np.array(sorted({tuple(sorted(pair)) for cell in cells
                             for pair in combinations(cell, 2)
                             if not set(pair) <= fixed}))
    vectors = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    squared_lengths = np.sum(vectors ** 2, axis=1)
    columns = np.full(16, -1)
    columns[free] = 3 * np.arange(len(free))
    return Geometry(positions, cells, triangles, pairs, squared_lengths,
                    free, columns, driven, hinge, leg, base)


def driven_configuration(geometry, angle):
    """Prescribe one neighbor's hinge angle while fixing the central unit."""
    positions = geometry.positions.copy()
    first, second = positions[geometry.hinge]
    transform = rotation(second - first, np.radians(angle))
    positions[geometry.driven] = (positions[geometry.driven] - first) @ transform.T + first
    return positions


def constraints(geometry, positions):
    vectors = positions[geometry.pairs[:, 0]] - positions[geometry.pairs[:, 1]]
    return (np.sum(vectors ** 2, axis=1) - geometry.squared_lengths) / geometry.leg ** 2


def constraint_jacobian(geometry, positions):
    first, second = geometry.pairs.T
    vectors = 2 * (positions[first] - positions[second]) / geometry.leg ** 2
    jacobian = np.zeros((len(first), 3 * len(geometry.free)))
    for sign, indices in ((1, first), (-1, second)):
        active = geometry.columns[indices] >= 0
        rows = np.flatnonzero(active)
        columns = geometry.columns[indices[active]]
        jacobian[rows[:, None], columns[:, None] + np.arange(3)] = sign * vectors[active]
    return jacobian


def vertex_chirality(positions):
    """Oriented crease-ray volumes at the four interior grid vertices."""
    signatures = []
    for row, column in ((1, 1), (1, 2), (2, 1), (2, 2)):
        index = 4 * row + column
        neighbors = [index - 4, index + 1, index + 4, index - 1]
        rays = positions[neighbors] - positions[index]
        signatures.append([
            np.linalg.det(rays[[(offset + step) % 4 for step in range(3)]])
            for offset in range(4)
        ])
    return np.array(signatures)


def probe_rotation(geometry, angle):
    """Attempt a prescribed hinge rotation without relaxing panel rigidity."""
    prescribed = driven_configuration(geometry, angle)

    def unpack(variables):
        positions = prescribed.copy()
        positions[geometry.free] = variables.reshape(-1, 3)
        return positions

    result = least_squares(
        lambda variables: constraints(geometry, unpack(variables)),
        geometry.positions[geometry.free].ravel(),
        jac=lambda variables: constraint_jacobian(geometry, unpack(variables)),
        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=120,
    )
    positions = unpack(result.x)
    residual = np.max(np.abs(constraints(geometry, positions)))
    panels = positions[geometry.cells]
    reference = geometry.positions[geometry.cells]
    lengths = np.linalg.norm(panels[:, :, None] - panels[:, None, :], axis=-1)
    reference_lengths = np.linalg.norm(reference[:, :, None] - reference[:, None, :], axis=-1)
    chirality = vertex_chirality(positions)
    reference_chirality = vertex_chirality(geometry.positions)
    preserved = np.array_equal(np.sign(chirality), np.sign(reference_chirality))
    preserved = bool(preserved and np.min(np.abs(chirality)) > 1e-8)
    return dict(angle_degrees=float(angle), normalized_squared_length_residual=float(residual),
                max_rigid_distance_error=float(np.max(np.abs(lengths - reference_lengths))),
                chirality_preserved=preserved,
                closed=bool(result.success and residual < 1e-9 and preserved))


def verify_source(filename):
    """Compare the selected patch directly against a downloaded GGB file."""
    with ZipFile(filename) as archive:
        root = ElementTree.fromstring(archive.read('geogebra.xml'))
    points = {}
    for element in root.findall('./construction/element'):
        kind = element.get('type')
        coordinates = element.find('coords')
        if kind not in ('point', 'point3d') or coordinates is None:
            continue
        if kind == 'point':
            scale = float(coordinates.get('z'))
            point = [float(coordinates.get('x')) / scale,
                     float(coordinates.get('y')) / scale, 0.0]
        else:
            scale = float(coordinates.get('w'))
            point = [float(coordinates.get(axis)) / scale for axis in ('x', 'y', 'z')]
        points[element.get('label')] = point
    reference = make_geometry(np.sqrt(6), 2)
    actual = np.array([points[label] for label in POINT_LABELS])
    np.testing.assert_allclose(actual - [0, 0, 0.5], reference.positions, atol=1e-12)
    polygons = {frozenset(command.find('input').attrib.values())
                for command in root.findall('./construction/command')
                if command.get('name') == 'Polygon'}
    for triangle in reference.triangles:
        labels = frozenset(POINT_LABELS[index] for index in triangle)
        if labels not in polygons:
            raise AssertionError(f'Triangle {sorted(labels)} does not match the source.')
    return 'All 16 vertices and 18 triangles match the GeoGebra 3x3 patch.'


def validate_geometry(geometry):
    """Check congruence, exact triangle dimensions, seams and nonplanarity."""
    panels = geometry.positions[geometry.cells]
    distances = np.linalg.norm(panels[:, :, None] - panels[:, None, :], axis=-1)
    np.testing.assert_allclose(distances, np.repeat(distances[:1], 9, axis=0), atol=1e-12)
    triangles = geometry.positions[geometry.triangles]
    lengths = np.sort(np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=-1), axis=1)
    np.testing.assert_allclose(lengths, np.tile([geometry.base, geometry.leg, geometry.leg], (18, 1)),
                               atol=1e-12)
    volumes = np.linalg.det(panels[:, 1:] - panels[:, :1])
    if np.min(np.abs(volumes)) < 1e-8:
        raise AssertionError('Every quadrilateral must be nonplanar.')
    edges = {}
    for cell in geometry.cells:
        for first, second in zip(cell, np.roll(cell, -1)):
            edge = tuple(sorted((int(first), int(second))))
            edges[edge] = edges.get(edge, 0) + 1
    if len(edges) != 24 or sum(count == 2 for count in edges.values()) != 12:
        raise AssertionError('Expected 24 grid edges, including 12 shared hinges.')
    if any(count not in (1, 2) for count in edges.values()):
        raise AssertionError('Unexpected edge incidence.')
    midpoint = (panels[:, 1] + panels[:, 3]) / 2
    first = panels[:, 0] - midpoint
    second = panels[:, 2] - midpoint
    cosine = np.sum(first * second, axis=1) / (
        np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1))
    dihedrals = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    np.testing.assert_allclose(dihedrals, dihedrals[0], atol=1e-10)
    return float(dihedrals[0])


def rigidity_audit(geometry):
    """Check the second-order solvability condition J*a + H[v,v] = 0.

    Pin the central rigid unit, leaving 36 position variables. A nonzero
    left-null projection of H[v,v] blocks continuation of the sole
    infinitesimal flex. Nullity one alone does not imply a finite motion.
    """
    fixed = set(geometry.cells[4])
    free = np.array([index for index in range(16) if index not in fixed])
    pairs = np.array(sorted({tuple(sorted(pair)) for cell in geometry.cells
                             for pair in combinations(cell, 2)
                             if not set(pair) <= fixed}))
    positions = geometry.positions / geometry.leg
    matrix = np.zeros((len(pairs), 3 * len(free)))
    columns = {int(index): 3 * offset for offset, index in enumerate(free)}
    for row, (first, second) in enumerate(pairs):
        direction = 2 * (positions[first] - positions[second])
        for sign, index in ((1, first), (-1, second)):
            if index in columns:
                offset = columns[index]
                matrix[row, offset:offset + 3] = sign * direction
    left, singular, right = np.linalg.svd(matrix)
    tolerance = singular[0] * 1e-10
    rank = int(np.count_nonzero(singular > tolerance))
    nullity = matrix.shape[1] - rank
    if nullity != 1:
        raise AssertionError(f'This audit expects one infinitesimal flex, found {nullity}.')
    velocity = np.zeros((16, 3))
    velocity[free] = right[-1].reshape(-1, 3)
    differences = velocity[pairs[:, 0]] - velocity[pairs[:, 1]]
    curvature = 2 * np.sum(differences ** 2, axis=1)
    obstruction = float(np.linalg.norm(left[:, rank:].T @ curvature))
    return dict(
        free_coordinates=int(matrix.shape[1]), constraints=int(matrix.shape[0]),
        jacobian_rank=rank, infinitesimal_modes=nullity,
        smallest_nonzero_singular_value=float(singular[rank - 1]),
        infinitesimal_residual=float(np.linalg.norm(matrix @ right[-1])),
        second_order_obstruction=obstruction,
        second_order_locked=bool(obstruction > 1e-8),
    )


def exact_reference_certificate():
    """Verify a rational second-order obstruction for the source coordinates.

    For integer source vertices, explicit rational vectors v,w satisfy
    J*v=0 and J.T*w=0, but w.T*H[v,v]=16. Products below are integers or
    binary-exact half-integers, so these equalities need no tolerance.
    """
    positions = np.array([[2 * (row + column - 3), column - row, (row + column + 1) % 2]
                          for row in range(4) for column in range(4)], dtype=float)
    cells = make_geometry().cells
    fixed = set(cells[4])
    free = [index for index in range(16) if index not in fixed]
    pairs = sorted({tuple(sorted(pair)) for cell in cells for pair in combinations(cell, 2)
                    if not set(pair) <= fixed})
    velocity = np.array([
        [0, 0, 0], [0, 1, 1], [0.5, 0, 1], [0, 0, 0],
        [0, 1, -1], [0, 0, 0], [0, 0, 0], [0.5, 0, -1],
        [-0.5, 0, -1], [0, 0, 0], [0, 0, 0], [0, -1, -1],
        [0, 0, 0], [-0.5, 0, 1], [0, -1, 1], [0, 0, 0],
    ])
    stress = np.array([0, 0, 0, 1, 1, -1, 0, 0, 0, -1, 1, 0, 0, -1, 1, 0,
                       0, -1, 0, 0, 1, -1, 0, 1, 0, -1, 0, -1, 0, -1, 0, 1, 0, 0, 1, 0])
    matrix = np.zeros((36, 36))
    curvature = []
    for row, (first, second) in enumerate(pairs):
        direction = 2 * (positions[first] - positions[second])
        for sign, index in ((1, first), (-1, second)):
            if index in free:
                offset = 3 * free.index(index)
                matrix[row, offset:offset + 3] = sign * direction
        relative = velocity[first] - velocity[second]
        curvature.append(2 * np.dot(relative, relative))
    np.testing.assert_array_equal(matrix @ velocity[free].ravel(), np.zeros(36))
    np.testing.assert_array_equal(matrix.T @ stress, np.zeros(36))
    obstruction = float(stress @ curvature)
    if obstruction != 16:
        raise AssertionError('Exact reference obstruction must equal 16.')
    return dict(leg_length='sqrt(6)', unnormalized_second_order_obstruction=16,
                velocity_norm_squared=13, stress_norm_squared=16)


def render_assembly(geometry, output, dihedral):
    """Draw a row-aligned 3x3 view and a separate oblique spatial view."""
    output = Path(output)
    if output.suffix.lower() != '.png':
        raise ValueError('Use a .png output: no rigid folding GIF exists for this assembly.')
    faces = geometry.positions[geometry.triangles]
    edges = sorted({tuple(sorted((int(first), int(second))))
                    for cell in geometry.cells for first, second in zip(cell, np.roll(cell, -1))})
    centers = geometry.positions[geometry.cells].mean(axis=1)
    lower = geometry.positions.min(axis=0)
    upper = geometry.positions.max(axis=0)
    span = upper - lower
    perspective = output.with_name(output.stem + '_perspective.png')
    views = (
        (output, GRID_ELEVATION, GRID_AZIMUTH, GRID_ROLL, '3 rows × 3 columns'),
        (perspective, VIEW_ELEVATION, VIEW_AZIMUTH, 0, 'Same assembly · oblique spatial view'),
    )
    for path, elevation, azimuth, roll, subtitle in views:
        figure = plt.figure(figsize=(8.5, 8.5), facecolor='#f5f7fb')
        axis = figure.add_axes([0.06, 0.16, 0.88, 0.72], projection='3d')
        axis.set_facecolor('#f5f7fb')
        mesh = Poly3DCollection(faces, facecolors='#6fa6cf', edgecolors='none',
                               linewidths=0, antialiaseds=False)
        axis.add_collection3d(mesh)
        for first, second in edges:
            segment = geometry.positions[[first, second]]
            axis.plot(*segment.T, color='#233e53', linewidth=1.7, zorder=5)
        for index, center in enumerate(centers):
            row, column = divmod(index, 3)
            axis.text(*center, f'{row + 1},{column + 1}', color='#183b56', fontsize=12,
                      ha='center', va='center', zorder=10,
                      bbox=dict(facecolor='#eef6fc', edgecolor='none', alpha=0.85, pad=2))
        axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
        axis.set_box_aspect(span, zoom=1)
        axis.set_proj_type('ortho')
        axis.view_init(elev=elevation, azim=azimuth, roll=roll)
        axis.set_axis_off()
        figure.canvas.draw()
        horizontal, vertical, _ = proj3d.proj_transform(*geometry.positions.T, axis.get_proj())
        pixels = axis.transData.transform(np.column_stack([horizontal, vertical]))
        origin = axis.transData.transform([0, 0])
        margin = 0.07 * min(axis.bbox.width, axis.bbox.height)
        available = np.array([origin[0] - axis.bbox.x0 - margin,
                              axis.bbox.x1 - origin[0] - margin,
                              origin[1] - axis.bbox.y0 - margin,
                              axis.bbox.y1 - origin[1] - margin])
        extent = np.array([origin[0] - pixels[:, 0].min(), pixels[:, 0].max() - origin[0],
                           origin[1] - pixels[:, 1].min(), pixels[:, 1].max() - origin[1]])
        axis.set_box_aspect(span, zoom=float(np.min(available / extent)))
        if path == output:
            horizontal, vertical, _ = proj3d.proj_transform(*centers.T, axis.get_proj())
            np.testing.assert_allclose(np.ptp(horizontal.reshape(3, 3), axis=0), 0, atol=1e-12)
            np.testing.assert_allclose(np.ptp(vertical.reshape(3, 3), axis=1), 0, atol=1e-12)
            if not (np.all(np.diff(horizontal.reshape(3, 3), axis=1) > 0)
                    and np.all(np.diff(vertical.reshape(3, 3), axis=0) < 0)):
                raise AssertionError('Grid must read left-to-right and top-to-bottom.')
        figure.suptitle('3 × 3 Spatial Quadrilaterals', fontsize=20, fontweight='bold', y=0.96)
        figure.text(0.5, 0.905, subtitle, ha='center', fontsize=13, color='#42596d')
        figure.text(0.5, 0.11, '9 rigid units · 12 shared edges · Labels: (row, column)',
                    ha='center', fontsize=11, color='#42596d')
        figure.text(0.5, 0.065, f'Internal seams hidden · Fixed internal dihedral: {dihedral:.2f}°',
                    ha='center', fontsize=10, color='#42596d')
        figure.savefig(path, dpi=160, facecolor=figure.get_facecolor())
        plt.close(figure)
    return perspective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leg-length', type=float, default=2.45)
    parser.add_argument('--base-length', type=float, default=2.0)
    parser.add_argument('--output', type=Path, default=OUTPUT, help='Initial assembly PNG.')
    parser.add_argument('--check-only', action='store_true', help='Run geometry and rigidity checks without writing files.')
    parser.add_argument('--reference-ggb', type=Path, help='Optional downloaded mj826xqp.ggb, for source verification.')
    args = parser.parse_args()
    geometry = make_geometry(args.leg_length, args.base_length)
    dihedral = validate_geometry(geometry)
    report = dict(source=SOURCE_URL, quadrilaterals=9, triangles=18, shared_edges=12,
                  leg_length=geometry.leg, base_length=geometry.base,
                  long_diagonal=2 * geometry.base, internal_dihedral_degrees=dihedral,
                  rigidity=rigidity_audit(geometry), exact_source=exact_reference_certificate(),
                  motion_request='Both directions, preserving rigidity and vertex chirality; no stacking target.',
                  initial_vertex_chirality=vertex_chirality(geometry.positions).tolist(),
                  rotation_probes=[probe_rotation(geometry, angle) for angle in (-1, -0.1, -0.01, 0, 0.01, 0.1, 1)])
    if args.reference_ggb:
        report['source_verification'] = verify_source(args.reference_ggb)
    print(f'PASS: 9 congruent nonplanar units; 18 triangles with sides {geometry.leg:g}, {geometry.leg:g}, {geometry.base:g}.')
    print(f'Internal dihedral: {dihedral:.6f} degrees; 12 shared edges exactly connected.')
    audit = report['rigidity']
    print(f'Pinned-center Jacobian rank: {audit["jacobian_rank"]}/{audit["free_coordinates"]}; '
          f'infinitesimal modes: {audit["infinitesimal_modes"]}.')
    print(f'Second-order obstruction: {audit["second_order_obstruction"]:.9f}; exact source certificate: 16 != 0.')
    for probe in report['rotation_probes']:
        print(f'Probe {probe["angle_degrees"]:+.2f} deg: closure residual '
              f'{probe["normalized_squared_length_residual"]:.3e}; '
              f'distance error={probe["max_rigid_distance_error"]:.3e}; '
              f'chirality preserved={probe["chirality_preserved"]}; closed={probe["closed"]}')
    if args.reference_ggb:
        print(report['source_verification'])
    if not audit['second_order_locked']:
        raise AssertionError('Unexpected audit result: recheck the reference assembly before animating.')
    report['local_feasible_rotation_range_degrees'] = [0.0, 0.0]
    print('LOCKED in both directions at the initial state; no stacking target was imposed.')
    print('Local feasible rotation interval: [0, 0] degrees. No folding GIF is generated.')
    if not args.check_only:
        perspective = render_assembly(geometry, args.output, dihedral)
        report_path = args.output.with_name(args.output.stem + '_report.json')
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'Assembly: {args.output.resolve()}')
        print(f'Perspective: {perspective.resolve()}')
        print(f'Audit report: {report_path.resolve()}')


if __name__ == '__main__':
    main()
