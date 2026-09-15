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
INITIAL_ANGLE = np.degrees(2 * np.arctan(np.sqrt(FOLD_RATIO)))
VIEW_ELEVATION = 40
VIEW_AZIMUTH = -85
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

    Rotate each old corner rhombus exactly 90 degrees about its centroid,
    then choose its nearest vertex as the central attachment. All five
    rhombi now have parallel corresponding edges in this template layout.
    Squares are placed in their zero-rotation reference planes only.
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


def fold_path(n_half=60, amax_deg=130.0, amin_deg=INITIAL_ANGLE):
    """Noncoplanar closed branch after quarter-turning the corner templates.

    The corner angle at each central vertex is now mu=lambda, instead of
    mu=180-lambda. For adjacent half-angle tangents x,y this requires
    cos(lambda) * (1 + (x*y)**2) - 2*x*y = 0.
    One closed branch is (x_T,x_L,x_B,x_R)=(t,q/t,-t,-q/t),
    q=2-sqrt(3). Reference rho=0 is the square's central-plane reference,
    not a realizable flat configuration of the assembled nine panels.
    """
    if n_half < 2 or not 0 < amin_deg < amax_deg < 180:
        raise ValueError('Need n_half >= 2 and 0 < min-angle < max-angle < 180 degrees.')
    progress = np.linspace(0, 1, n_half + 1)
    drive = np.radians(amin_deg + (amax_deg - amin_deg)
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


def spatial_metrics(panels):
    """Height and smaller angles between square planes and the central plane."""
    normals = np.cross(panels[:, 1] - panels[:, 0], panels[:, 3] - panels[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    tilts = np.degrees(np.arccos(np.clip(np.abs(normals[1:5] @ normals[0]), 0, 1)))
    return np.ptp(panels[:, :, 2]), tilts


def audit_branches():
    """Enumerate regular labelled algebraic branches, modulo rho -> -rho.

    Products p_i=x_i*x_(i+1) have roots (+q,-q,+q,-q) or their
    corresponding reciprocals. Loop closure requires p_0*p_2=p_1*p_3.
    This counts algebraic assembly branches, not collision-free global
    motions or equivalence classes under relabelling the panels.
    """
    drive = np.sqrt(FOLD_RATIO)
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
    print('The displayed initial assembly selects the (+1,+1,+1,+1) branch. '
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


def validate(path):
    """Check panel shape, rigidity, planarity, seams and sampled crossings."""
    if len(SHARED_EDGES) != 12:
        raise AssertionError(f'Expected 12 shared edges, got {len(SHARED_EDGES)}')
    for index, panel in enumerate(TEMPLATES):
        lengths, angles = face_stats(panel)
        expected = [90] * 4 if 1 <= index <= 4 else [60, 60, 120, 120]
        np.testing.assert_allclose(lengths, 1, atol=1e-10)
        np.testing.assert_allclose(sorted(angles), expected, atol=1e-10)
        if index >= 5:
            np.testing.assert_allclose(angles[0], face_stats(TEMPLATES[0])[1][index - 5], atol=1e-10)
            central_edges = np.roll(TEMPLATES[0], -1, axis=0) - TEMPLATES[0]
            corner_edges = np.roll(panel, -1, axis=0) - panel
            np.testing.assert_allclose(np.max(np.abs(corner_edges @ central_edges.T), axis=1),
                                       1, atol=1e-10)
        print(f'{FACE_NAMES[index]:>2}: sides={np.round(lengths, 3)}, angles={np.round(angles, 1)}')
    reference_distances = np.linalg.norm(
        TEMPLATES[:, :, None, :] - TEMPLATES[:, None, :, :], axis=-1)
    vertex_cosines = np.cos(np.radians(face_stats(TEMPLATES[0])[1]))
    worst = dict(rigidity=0.0, angles=0.0, gap=0.0, registration=0.0,
                 planarity=0.0, hinge_closure=0.0)
    for rotations in path:
        radians = np.radians(rotations)
        following_radians = np.roll(radians, -1)
        equations = (-vertex_cosines * np.cos(radians) * np.cos(following_radians)
                     + np.sin(radians) * np.sin(following_radians) - vertex_cosines)
        worst['hinge_closure'] = max(worst['hinge_closure'], np.max(np.abs(equations)))
        jacobian = np.zeros((4, 4))
        for index in range(4):
            following = (index + 1) % 4
            first_angle, second_angle = radians[index], radians[following]
            jacobian[index, index] = (vertex_cosines[index] * np.sin(first_angle) * np.cos(second_angle)
                                      + np.cos(first_angle) * np.sin(second_angle))
            jacobian[index, following] = (vertex_cosines[index] * np.cos(first_angle) * np.sin(second_angle)
                                          + np.sin(first_angle) * np.cos(second_angle))
        if np.linalg.matrix_rank(jacobian, tol=1e-9) != 3:
            raise AssertionError('Expected one local DOF at this regular configuration.')
        current, residual = folded_faces(rotations)
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
    print(f'PASS: {len(path)} states, 9 rigid panels, 12 shared edges; no transverse crossings detected.')
    return worst


def render(path, output, fps=25):
    """Show the connected initial assembly and its motion with one camera."""
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
                'Initial configuration held fixed for comparison', ha='center',
                fontsize=10, color='#46566a', linespacing=1.6)
    state_text = figure.text(0.755, 0.10, '', ha='center', fontsize=10,
                             color='#46566a', linespacing=1.6)
    figure.text(0.5, 0.025, 'Blue: all 5 rhombi (60° / 120°) · Orange: 4 squares · '
                f'Same {VIEW_ELEVATION}° view and scale · Central panel fixed', ha='center',
                fontsize=10, color='#46566a')
    figure.subplots_adjust(left=0.035, right=0.99, bottom=0.17, top=0.84)

    def update(frame_index):
        angles = frames[frame_index]
        panels, _ = folded_faces(angles)
        height, tilts = spatial_metrics(panels)
        mesh.set_verts(panels)
        title.set_text('Fold / unfold\n' + '   '.join(
            f'{name}: {angle:+.0f}°' for name, angle in zip(FACE_NAMES[1:5], angles)))
        state_text.set_text(f'Height span: {height:.3f} (initial: {np.ptp(configurations[0, :, :, 2]):.3f})\n'
                            'Plane tilts to C (T/L/B/R): ' + ' / '.join(f'{tilt:.1f}°' for tilt in tilts))
        return mesh, title, state_text

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
    parser.add_argument('--min-angle', type=float, default=INITIAL_ANGLE,
                        help='Initial signed T rotation in degrees (default about 54.736).')
    parser.add_argument('--max-angle', type=float, default=130,
                        help='Final signed T rotation in degrees (<180; default 130).')
    parser.add_argument('--steps', type=int, default=60, help='Steps in each half cycle (>=2).')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check-only', action='store_true', help='Validate geometry without writing images.')
    parser.add_argument('--audit-branches', action='store_true',
                        help='Check regular algebraic assembly branches without writing images.')
    args = parser.parse_args()
    if args.audit_branches:
        audit_branches()
        return
    path = fold_path(args.steps, args.max_angle, args.min_angle)
    validate(path)
    if not args.check_only:
        render(path, args.output)


if __name__ == '__main__':
    main()
