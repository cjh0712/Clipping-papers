"""Spatial 3x3 folding with quarter-turned corner rhombi and hinged squares."""

import argparse
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


FACE_NAMES = ['C', 'T', 'L', 'B', 'R', 'TL', 'BL', 'BR', 'TR']
COLORS = ['#5294cc'] + ['#efa64b'] * 4 + ['#5294cc'] * 4
OUTPUT = Path(__file__).with_name('origami_grid3x3_rhombi_squares.gif')
FOLD_RATIO = 2 - np.sqrt(3)
START_ANGLE = np.degrees(2 * np.arctan(np.sqrt(FOLD_RATIO)))
END_ANGLE = 130.0
VIEW_ELEVATION = 40
VIEW_AZIMUTH = -25
VIEW_ZOOM = 1.16


def norm(vector):
    """Normalize a nonzero vector."""
    return vector / np.linalg.norm(vector)


def rotation(axis, angle):
    """Right-handed rotation, with angle in radians."""
    axis = norm(np.asarray(axis, float))
    skew = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
    return (np.cos(angle) * np.eye(3)
            + (1 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * skew)


def frame(first, second):
    """Orthonormal frame; mapping between frames preserves panel shape."""
    tangent = norm(first)
    normal = norm(np.cross(first, second))
    return np.column_stack([tangent, np.cross(normal, tangent), normal])


def panel_templates():
    """Individual planar templates, not a connected coplanar configuration.

    Rotate each old corner rhombus exactly 90 degrees about its centroid.
    Attach the nearest vertex after rotating, exchanging acute and obtuse
    attachment angles relative to the complementary-angle assembly.
    All five rhombi have parallel corresponding edges in template space.
    """
    down = np.array([0.0, -1.0, 0.0])
    right = np.array([np.sqrt(3) / 2, -0.5, 0.0])
    central = np.array([np.zeros(3), down, down + right, right])
    central -= central.mean(axis=0)
    panels = [central]
    hinges = []
    for index in range(4):
        start, end = central[index], central[(index - 1) % 4]
        direction = end - start
        outward = np.array([-direction[1], direction[0], 0.0])
        panels.append(np.array([start, end, end + outward, start + outward]))
        hinges.append((start, norm(direction)))
    quarter_turn = rotation([0, 0, 1], np.pi / 2)
    for index in range(4):
        vertex = central[index]
        first_face, second_face = index + 1, (index + 1) % 4 + 1
        first, second = panels[first_face][3], panels[second_face][2]
        corner = np.array([vertex, first, first + second - vertex, second])
        centroid = corner.mean(axis=0)
        corner = (corner - centroid) @ quarter_turn.T + centroid
        attachment = int(np.argmin(np.linalg.norm(corner - vertex, axis=1)))
        panels.append(np.roll(corner, -attachment, axis=0))
    return np.array(panels), hinges


TEMPLATES, HINGES = panel_templates()


def shared_edges():
    """Twelve shared edges specified by topology, not by planar proximity."""
    matches = []
    for index in range(4):
        first_face, second_face = index + 1, (index + 1) % 4 + 1
        matches.append((0, [index, (index - 1) % 4], first_face, [0, 1]))
        matches.append((index + 5, [0, 1], first_face, [0, 3]))
        matches.append((index + 5, [0, 3], second_face, [1, 2]))
    return matches


SHARED_EDGES = shared_edges()


def fold_path(n_half=60, start_deg=START_ANGLE, end_deg=END_ANGLE):
    """Fold downward on one noncoplanar branch without changing chirality.

    At each vertex the corner angle is mu=lambda. For adjacent
    half-angle tangents x,y, compatibility is
    cos(lambda)*(1+(x*y)**2)-2*x*y=0. One closed branch is
    (x_T,x_L,x_B,x_R)=(t,q/t,-t,-q/t), q=2-sqrt(3).
    Both endpoints are noncoplanar; the path does not cross t=0.
    """
    if n_half < 2 or not 0 < start_deg < end_deg < 180:
        raise ValueError('Need n_half >= 2 and 0 < start-angle < end-angle < 180 degrees.')
    progress = np.linspace(0, 1, n_half + 1)
    drive = np.radians(start_deg + (end_deg - start_deg)
                       * (1 - np.cos(np.pi * progress)) / 2)
    tangent = np.tan(drive / 2)
    half_tangents = np.column_stack([tangent, FOLD_RATIO / tangent,
                                    -tangent, -FOLD_RATIO / tangent])
    return np.degrees(2 * np.arctan(half_tangents))


def folded_faces(rotations):
    """Rotate squares around C and rigidly place each corner rhombus."""
    current = [TEMPLATES[0].copy()]
    for index, (origin, axis) in enumerate(HINGES):
        transform = rotation(axis, np.radians(rotations[index]))
        current.append((TEMPLATES[index + 1] - origin) @ transform.T + origin)
    residual = 0.0
    for index in range(4):
        first_face, second_face = index + 1, (index + 1) % 4 + 1
        origin = TEMPLATES[0, index]
        original = TEMPLATES[index + 5]
        source_origin = original[0]
        first_target, second_target = current[first_face][3], current[second_face][2]
        original_frame = frame(original[1] - source_origin, original[3] - source_origin)
        target_frame = frame(first_target - origin, second_target - origin)
        transform = target_frame @ original_frame.T
        corner = (original - source_origin) @ transform.T + origin
        residual = max(residual, np.linalg.norm(corner[1] - first_target),
                       np.linalg.norm(corner[3] - second_target),
                       np.max(np.abs(transform.T @ transform - np.eye(3))))
        current.append(corner)
    return np.array(current), residual


def face_stats(points):
    """Cyclic edge lengths and interior angles in degrees."""
    previous = np.roll(points, 1, axis=0) - points
    following = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(following, axis=1)
    cosines = np.sum(previous * following, axis=1) / (
        np.linalg.norm(previous, axis=1) * lengths)
    return lengths, np.degrees(np.arccos(np.clip(cosines, -1, 1)))


def grid_vertices(panels):
    """Recover the 4x4 shared vertices in row/column order."""
    mappings = ((5, [2, 3, 0, 1]), (1, [3, 0, 1, 2]), (8, [3, 0, 1, 2]),
                (2, [2, 3, 0, 1]), (0, [0, 1, 2, 3]), (4, [0, 1, 2, 3]),
                (6, [1, 2, 3, 0]), (3, [1, 2, 3, 0]), (7, [0, 1, 2, 3]))
    faces = np.array([panels[index, order] for index, order in mappings]).reshape(3, 3, 4, 3)
    vertices = np.full((4, 4, 3), np.nan)
    for row in range(3):
        for column in range(3):
            for corner, (row_offset, column_offset) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                target = vertices[row + row_offset, column + column_offset]
                point = faces[row, column, corner]
                if np.isfinite(target).all():
                    np.testing.assert_allclose(target, point, atol=1e-9, rtol=0)
                target[:] = point
    return vertices


def row_column_spans(vertices):
    """Straight spatial distances along one row and one column, not screen bounds."""
    vertices = np.asarray(vertices, dtype=float)
    return np.linalg.norm([vertices[0, -1] - vertices[0, 0],
                           vertices[-1, 0] - vertices[0, 0]], axis=1)


def contraction_report(spans, require_isotropic=False):
    """Audit equal fractional contraction separately from rigid-fold closure."""
    spans = np.asarray(spans, dtype=float)
    if spans.ndim != 2 or spans.shape[1] != 2 or len(spans) < 2:
        raise ValueError('Need at least two pairs of row/column spans.')
    if not np.isfinite(spans).all() or np.any(spans <= 1e-9):
        raise ValueError('Cannot normalize a zero or invalid spatial span.')
    ratios = spans / spans[0]
    mismatch = float(np.max(np.abs(ratios[:, 0] - ratios[:, 1])))
    monotone = bool(np.all(np.diff(ratios, axis=0) <= 1e-9))
    contracting = bool(np.all(ratios[-1] < 1 - 1e-9))
    isotropic = mismatch <= 1e-9 and monotone and contracting
    for index, label in enumerate(('along a row (H)', 'along a column (V)')):
        print(f'Spatial span {label}: {spans[0, index]:.6f} -> {spans[-1, index]:.6f}; '
              f'change = {100 * (ratios[-1, index] - 1):+.3f}%')
    print(f'Max H/V normalized-span difference = {mismatch:.6e}')
    if not isotropic:
        message = ('This path is NOT equal biaxial contraction. Rigid panels and closed seams '
                   'do not imply isotropic shrinkage; no display rescaling is applied.')
        if require_isotropic:
            raise ValueError(message)
        print('WARNING: ' + message)
    else:
        print('PASS: both spatial spans contract by the same fraction at every sampled state.')
    return dict(ratios=ratios, mismatch=mismatch, isotropic=isotropic)


def spatial_metrics(panels):
    """Height and smaller angles between square planes and the central plane."""
    normals = np.cross(panels[:, 1] - panels[:, 0], panels[:, 3] - panels[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    tilts = np.degrees(np.arccos(np.clip(np.abs(normals[1:5] @ normals[0]), 0, 1)))
    return np.ptp(panels[:, :, 2]), tilts


def long_diagonals(panels):
    """Endpoints of each rhombus's long diagonal in assembled coordinates."""
    result = []
    for panel in panels[[0, 5, 6, 7, 8]]:
        pairs = np.array([panel[[0, 2]], panel[[1, 3]]])
        lengths = np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1)
        result.append(pairs[np.argmax(lengths)])
    return np.array(result)


def rhombus_direction_error(panels):
    """Compare long axes after the shortest rotation aligning face normals."""
    diagonals = long_diagonals(panels)
    directions = diagonals[:, 1] - diagonals[:, 0]
    central_normal = norm(np.cross(panels[0, 1] - panels[0, 0], panels[0, 3] - panels[0, 0]))
    central_direction = norm(directions[0])
    errors = []
    for panel, direction in zip(panels[5:], directions[1:]):
        normal = norm(np.cross(panel[1] - panel[0], panel[3] - panel[0]))
        axis = np.cross(normal, central_normal)
        cosine = np.clip(np.dot(normal, central_normal), -1, 1)
        if np.linalg.norm(axis) < 1e-10:
            if cosine < 0:
                raise AssertionError('Antiparallel normals make tilt removal ambiguous.')
            aligned = norm(direction)
        else:
            aligned = rotation(axis, np.arccos(cosine)) @ norm(direction)
        errors.append(np.linalg.norm(np.cross(aligned, central_direction)))
    return max(errors)


def vertex_chirality(panels):
    """Signed volumes of consecutive crease-ray triples at four vertices."""
    signatures = []
    for index in range(4):
        vertex = panels[0, index]
        first_face = index + 1
        second_face = (index + 1) % 4 + 1
        rays = np.array([
            panels[0, (index - 1) % 4] - vertex,
            panels[first_face, 3] - vertex,
            panels[second_face, 2] - vertex,
            panels[0, (index + 1) % 4] - vertex,
        ])
        signatures.append([
            np.linalg.det(np.stack([rays[offset], rays[(offset + 1) % 4],
                                    rays[(offset + 2) % 4]]))
            for offset in range(4)
        ])
    return np.array(signatures)


def audit_branches():
    """Enumerate regular labelled algebraic branches, modulo rho -> -rho.

    Products p_i=x_i*x_(i+1) have roots (+q,-q,+q,-q), with q optionally
    replaced by 1/q. Loop closure requires p_0*p_2=p_1*p_3.
    This counts algebraic assembly branches, not collision-free global
    motions or equivalence classes under relabelling the panels.
    """
    drive = np.tan(np.radians(70) / 2)
    reference_distances = np.linalg.norm(
        TEMPLATES[:, :, None, :] - TEMPLATES[:, None, :, :], axis=-1)
    count = 0
    for powers in product([-1, 1], repeat=4):
        if powers[0] + powers[2] != powers[1] + powers[3]:
            continue
        products = np.array([1, -1, 1, -1]) * FOLD_RATIO ** np.array(powers)
        tangents = np.array([drive, products[0] / drive,
                             products[1] * drive / products[0],
                             products[0] * products[2] / (products[1] * drive)])
        angles = np.degrees(2 * np.arctan(tangents))
        panels, residual = folded_faces(angles)
        distances = np.linalg.norm(panels[:, :, None, :] - panels[:, None, :, :], axis=-1)
        np.testing.assert_allclose(distances, reference_distances, atol=1e-10)
        np.testing.assert_allclose(tangents * np.roll(tangents, -1), products, atol=1e-10)
        if residual > 1e-10:
            raise AssertionError(f'Branch {powers} fails geometric closure.')
        jacobian = np.zeros((4, 4))
        for index in range(4):
            following = (index + 1) % 4
            jacobian[index, index] = tangents[following]
            jacobian[index, following] = tangents[index]
        rank = np.linalg.matrix_rank(jacobian)
        if rank != 3:
            raise AssertionError(f'Unexpected branch Jacobian rank: {rank}')
        count += 1
        print(f'branch {count}: root powers={powers}; rho(T,L,B,R)={np.round(angles, 3)}; '
              f'local DOF={4 - rank}; closure={residual:.2e}')
    print(f'{count} labelled algebraic branches modulo global sign reversal; '
          'each regular branch has one local DOF.')
    print('The displayed assembly selects the (+1,+1,+1,+1) branch. '
          'This is not a global collision/branch-switching classification.')
    return count


def pierces_panel(edges, panel, tolerance=1e-9):
    """Detect transverse edge crossings strictly inside a convex panel."""
    normal = norm(np.cross(panel[1] - panel[0], panel[3] - panel[0]))
    distances = (edges - panel[0]) @ normal
    for index in range(4):
        following = (index + 1) % 4
        if distances[index] * distances[following] >= -tolerance**2:
            continue
        fraction = distances[index] / (distances[index] - distances[following])
        point = edges[index] + fraction * (edges[following] - edges[index])
        inside = [np.dot(np.cross(panel[(side + 1) % 4] - panel[side],
                                   point - panel[side]), normal) for side in range(4)]
        if min(inside) > tolerance:
            return True
    return False


def validate(path, require_isotropic=False):
    """Check panel shape, rigidity, planarity, seams and sampled crossings."""
    if len(SHARED_EDGES) != 12:
        raise AssertionError(f'Expected 12 shared edges, got {len(SHARED_EDGES)}')
    for index, panel in enumerate(TEMPLATES):
        lengths, angles = face_stats(panel)
        expected = [90] * 4 if 1 <= index <= 4 else [60, 60, 120, 120]
        np.testing.assert_allclose(lengths, 1, atol=1e-10)
        np.testing.assert_allclose(sorted(angles), expected, atol=1e-10)
        if index >= 5:
            central_angle = face_stats(TEMPLATES[0])[1][index - 5]
            np.testing.assert_allclose(angles[0], central_angle, atol=1e-10)
            central_edges = np.roll(TEMPLATES[0], -1, axis=0) - TEMPLATES[0]
            corner_edges = np.roll(panel, -1, axis=0) - panel
            np.testing.assert_allclose(np.max(np.abs(corner_edges @ central_edges.T), axis=1),
                                       1, atol=1e-10)
        print(f'{FACE_NAMES[index]:>2}: sides={np.round(lengths, 3)}, angles={np.round(angles, 1)}')
    reference_distances = np.linalg.norm(
        TEMPLATES[:, :, None, :] - TEMPLATES[:, None, :, :], axis=-1)
    vertex_cosines = np.cos(np.radians(face_stats(TEMPLATES[0])[1]))
    initial_panels, _ = folded_faces(path[0])
    initial_chirality = vertex_chirality(initial_panels)
    if np.min(np.abs(initial_chirality)) < 1e-8:
        raise AssertionError('Initial configuration is too close to a vertex inversion.')
    chirality_signs = np.sign(initial_chirality)
    worst = dict(rigidity=0.0, angles=0.0, gap=0.0, registration=0.0,
                 planarity=0.0, hinge_closure=0.0, rhombus_direction=0.0,
                 chirality_margin=np.inf)
    for rotations in path:
        half_tangents = np.tan(np.radians(rotations) / 2)
        following_tangents = np.roll(half_tangents, -1)
        products = half_tangents * following_tangents
        equations = vertex_cosines * (1 + products ** 2) - 2 * products
        worst['hinge_closure'] = max(worst['hinge_closure'], np.max(np.abs(equations)))
        jacobian = np.zeros((4, 4))
        for index in range(4):
            following = (index + 1) % 4
            factor = 2 * vertex_cosines[index] * products[index] - 2
            jacobian[index, index] = factor * half_tangents[following]
            jacobian[index, following] = factor * half_tangents[index]
        if np.linalg.matrix_rank(jacobian, tol=1e-9) != 3:
            raise AssertionError('Expected one local DOF at this regular configuration.')
        current, residual = folded_faces(rotations)
        worst['rhombus_direction'] = max(worst['rhombus_direction'], rhombus_direction_error(current))
        chirality = vertex_chirality(current)
        if not np.array_equal(np.sign(chirality), chirality_signs):
            raise AssertionError(f'Vertex concavity changed at rotations {rotations}.')
        worst['chirality_margin'] = min(worst['chirality_margin'], np.min(np.abs(chirality)))
        worst['registration'] = max(worst['registration'], residual)
        distances = np.linalg.norm(current[:, :, None, :] - current[:, None, :, :], axis=-1)
        worst['rigidity'] = max(worst['rigidity'], np.max(np.abs(distances - reference_distances)))
        for index, panel in enumerate(current):
            worst['angles'] = max(worst['angles'], np.max(np.abs(
                face_stats(panel)[1] - face_stats(TEMPLATES[index])[1])))
            normal = norm(np.cross(panel[1] - panel[0], panel[3] - panel[0]))
            worst['planarity'] = max(worst['planarity'], abs(np.dot(panel[2] - panel[0], normal)))
        for first_face, first_pair, second_face, second_pair in SHARED_EDGES:
            worst['gap'] = max(worst['gap'], np.max(np.linalg.norm(
                current[first_face, first_pair] - current[second_face, second_pair], axis=1)))
        for first_face in range(9):
            for second_face in range(first_face + 1, 9):
                if (pierces_panel(current[first_face], current[second_face])
                        or pierces_panel(current[second_face], current[first_face])):
                    raise AssertionError(f'Panels {FACE_NAMES[first_face]}/{FACE_NAMES[second_face]}'
                                         f' intersect at rotations {rotations}')
    for name, error in worst.items():
        if name == 'chirality_margin':
            print(f'min {name} = {error:.3e}')
            if error < 1e-8:
                raise AssertionError(f'{name} validation failed: {error}')
        else:
            print(f'max {name} error = {error:.3e}')
            if error > 1e-8:
                raise AssertionError(f'{name} validation failed: {error}')
    for label, rotations in [('initial', path[0]), ('final', path[-1])]:
        panels, _ = folded_faces(rotations)
        height, tilts = spatial_metrics(panels)
        seam_angles = []
        for first_face, _, second_face, _ in SHARED_EDGES:
            first_panel, second_panel = panels[first_face], panels[second_face]
            first_normal = norm(np.cross(first_panel[1] - first_panel[0], first_panel[3] - first_panel[0]))
            second_normal = norm(np.cross(second_panel[1] - second_panel[0], second_panel[3] - second_panel[0]))
            seam_angles.append(np.degrees(np.arccos(np.clip(abs(np.dot(first_normal, second_normal)), 0, 1))))
        print(f'{label}: height span = {height:.6f}; min shared-plane angle = {min(seam_angles):.3f} deg')
        print(f'{label}: square-plane tilts to C (T,L,B,R) = {np.round(tilts, 3)} deg')
        if height < 1e-8 or min(seam_angles) < 1e-8:
            raise AssertionError(f'{label} must be a noncoplanar assembly.')
    initial_height = spatial_metrics(folded_faces(path[0])[0])[0]
    final_height = spatial_metrics(folded_faces(path[-1])[0])[0]
    print(f'Height change = {100 * (final_height / initial_height - 1):+.3f}%; '
          'physical constraint Jacobian rank = 3 (one local DOF).')
    print('Vertex chirality signs stayed fixed at all four intersections.')
    print(f'PASS: {len(path)} states, 9 rigid panels, 12 shared edges; no transverse crossings detected.')
    contraction_report([row_column_spans(grid_vertices(folded_faces(angles)[0]))
                        for angles in path], require_isotropic)
    return worst


def render(path, output, fps=25):
    """Show both assemblies with the same fixed camera and physical scale."""
    hold = max(1, fps // 2)
    frames = np.concatenate([np.repeat(path[:1], hold, axis=0), path[1:],
                             np.repeat(path[-1:], hold, axis=0), path[-2:0:-1]])
    figure = plt.figure(figsize=(13, 7), facecolor='#f5f7fb')
    layout = figure.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.04)
    assembly_axis = figure.add_subplot(layout[0], projection='3d')
    fold_axis = figure.add_subplot(layout[1], projection='3d')
    configurations = np.array([folded_faces(angles)[0] for angles in path])
    initial_mesh = Poly3DCollection(configurations[0], facecolors=COLORS,
                                    edgecolors='#203349', linewidths=1.2)
    assembly_axis.add_collection3d(initial_mesh)
    mesh = Poly3DCollection(configurations[0], facecolors=COLORS, edgecolors='#203349', linewidths=1.2)
    fold_axis.add_collection3d(mesh)
    diagonal_lines = []
    for diagonal in long_diagonals(configurations[0]):
        assembly_axis.plot(*diagonal.T, color='#173c63', linestyle='--', linewidth=1.4, zorder=10)
        line, = fold_axis.plot(*diagonal.T, color='#173c63', linestyle='--', linewidth=1.4, zorder=10)
        diagonal_lines.append(line)
    lower = configurations.min(axis=(0, 1, 2))
    upper = configurations.max(axis=(0, 1, 2))
    padding = (upper - lower) * 0.06
    for axis in (assembly_axis, fold_axis):
        axis.set(xlim=(lower[0] - padding[0], upper[0] + padding[0]),
                 ylim=(lower[1] - padding[1], upper[1] + padding[1]),
                 zlim=(lower[2] - padding[2], upper[2] + padding[2]))
        axis.set_box_aspect(upper - lower, zoom=VIEW_ZOOM)
        axis.set_proj_type('ortho')
        axis.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
        axis.set_axis_off()
        axis.set_facecolor('#f5f7fb')
    initial_angles = '   '.join(f'{name}: {angle:+.0f}°'
                              for name, angle in zip(FACE_NAMES[1:5], path[0]))
    assembly_axis.set_title('Initial 3D assembly (fixed)\n' + initial_angles, fontsize=12, pad=0)
    title = fold_axis.set_title('', fontsize=12, pad=0)
    figure.suptitle('3 × 3 Spatial Folding · Rhombi + Squares',
                   fontsize=16, fontweight='bold', y=0.97)
    figure.text(0.275, 0.10, '9 connected panels · 12 shared edges\n'
                'Four vertex-chirality signs fixed during folding', ha='center',
                fontsize=10, color='#46566a', linespacing=1.6)
    state_text = figure.text(0.755, 0.10, '', ha='center', fontsize=10,
                             color='#46566a', linespacing=1.6)
    figure.text(0.5, 0.025, 'Blue: 5 rhombi · Orange: 4 squares · Dashed: long diagonals · '
                f'Same {VIEW_ELEVATION}° camera · Same fixed scale in both views', ha='center',
                fontsize=10, color='#46566a')
    figure.subplots_adjust(left=0.035, right=0.99, bottom=0.17, top=0.84)

    def update(frame_index):
        angles = frames[frame_index]
        panels, _ = folded_faces(angles)
        height, tilts = spatial_metrics(panels)
        mesh.set_verts(panels)
        for line, diagonal in zip(diagonal_lines, long_diagonals(panels)):
            line.set_data_3d(*diagonal.T)
        title.set_text('Fold downward / return\n' + '   '.join(
            f'{name}: {angle:+.0f}°' for name, angle in zip(FACE_NAMES[1:5], angles)))
        state_text.set_text(f'Height span: {height:.3f} (initial: {np.ptp(configurations[0, :, :, 2]):.3f})\n'
                            'Plane tilts to C (T/L/B/R): ' + ' / '.join(f'{tilt:.1f}°' for tilt in tilts)
                            + ' · chirality fixed')
        return mesh, title, state_text, *diagonal_lines

    update(0)
    assembled = output.with_name(output.stem + '_assembled.png')
    figure.savefig(assembled, dpi=150, facecolor=figure.get_facecolor())
    animation = FuncAnimation(figure, update, frames=len(frames), interval=1000 / fps, blit=False)
    animation.save(str(output), writer=PillowWriter(fps=fps), dpi=100)
    update(hold + len(path) - 2)
    preview = output.with_suffix('.png')
    figure.savefig(preview, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f'GIF: {output}\nPreview: {preview}\nInitial assembly: {assembled}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-angle', type=float, default=START_ANGLE,
                        help='Initial signed T rotation in degrees (default about 54.736).')
    parser.add_argument('--end-angle', type=float, default=END_ANGLE,
                        help='Final signed T rotation in degrees (default 130).')
    parser.add_argument('--steps', type=int, default=60, help='Steps in each half cycle (>=2).')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check-only', action='store_true', help='Validate geometry without writing images.')
    parser.add_argument('--audit-branches', action='store_true',
                        help='Check regular algebraic assembly branches without writing images.')
    parser.add_argument('--require-isotropic', action='store_true',
                        help='Reject paths without equal fractional H/V contraction before writing images.')
    args = parser.parse_args()
    if args.audit_branches:
        audit_branches()
        return
    path = fold_path(args.steps, args.start_angle, args.end_angle)
    try:
        validate(path, args.require_isotropic)
    except ValueError as error:
        parser.error(str(error))
    if not args.check_only:
        render(path, args.output)


if __name__ == '__main__':
    main()
