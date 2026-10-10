"""Periodic 5x5 extension of the planar rhombus-and-square folding model."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import origami_grid3x3 as base_model


SIZE = 5
OUTPUT = Path(__file__).with_name('origami_grid5x5_rhombi_squares.gif')
VIEW_ELEVATION = 40
VIEW_AZIMUTH = -25
VIEW_ROLL = 85
RHOMBI = np.array([(row + column) % 2 == 0 for row in range(SIZE) for column in range(SIZE)])
COLORS = np.where(RHOMBI, '#5294cc', '#efa64b').tolist()
VERTEX_IDS = np.arange((SIZE + 1) ** 2).reshape(SIZE + 1, SIZE + 1)
FACE_VERTICES = np.stack([VERTEX_IDS[:-1, :-1], VERTEX_IDS[1:, :-1],
                          VERTEX_IDS[1:, 1:], VERTEX_IDS[:-1, 1:]], axis=2).reshape(-1, 4)


def shared_edges():
    """Collect face-local endpoints for each of the forty common edges."""
    incidences = {}
    for face_index, indices in enumerate(FACE_VERTICES):
        for corner in range(4):
            following = (corner + 1) % 4
            first, second = int(indices[corner]), int(indices[following])
            local = [corner, following] if first < second else [following, corner]
            incidences.setdefault(tuple(sorted((first, second))), []).append((face_index, local))
    if len(incidences) != 60 or any(len(matches) not in (1, 2) for matches in incidences.values()):
        raise AssertionError('Expected sixty grid edges with at most two incident faces.')
    return [matches for matches in incidences.values() if len(matches) == 2]


SHARED_EDGES = shared_edges()


def grid_configuration(rotations):
    """Repeat the two row/column edge vectors of the original 3x3.

    Each face is a parallelogram, so P[row,column]=origin+R[row]+C[column].
    Alternating unit edge vectors have mutual dot products 1/2 or 0,
    giving 60/120-degree rhombi or squares. The central old 3x3 remains
    at exactly its original coordinates for every drive angle.
    """
    seed, residual = base_model.folded_faces(rotations)
    if residual > 1e-9:
        raise AssertionError('The seed 3x3 failed closure.')
    central = seed[0]
    row_even = central[1] - central[0]
    row_odd = central[0] - seed[1, 3]
    column_even = central[3] - central[0]
    column_odd = central[0] - seed[2, 2]
    row_steps = np.array([row_even if index % 2 == 0 else row_odd for index in range(SIZE)])
    column_steps = np.array([column_even if index % 2 == 0 else column_odd for index in range(SIZE)])
    row_offsets = np.vstack([np.zeros(3), np.cumsum(row_steps, axis=0)])
    column_offsets = np.vstack([np.zeros(3), np.cumsum(column_steps, axis=0)])
    middle = SIZE // 2
    origin = central[0] - row_offsets[middle] - column_offsets[middle]
    vertices = origin + row_offsets[:, None] + column_offsets[None, :]
    faces = vertices.reshape(-1, 3)[FACE_VERTICES]
    return vertices, faces


def central_reference(rotations):
    """Original nine faces, ordered by grid row and canonical corners."""
    seed, _ = base_model.folded_faces(rotations)
    mappings = ((5, [2, 3, 0, 1]), (1, [3, 0, 1, 2]), (8, [3, 0, 1, 2]),
                (2, [2, 3, 0, 1]), (0, [0, 1, 2, 3]), (4, [0, 1, 2, 3]),
                (6, [1, 2, 3, 0]), (3, [1, 2, 3, 0]), (7, [0, 1, 2, 3]))
    return np.array([seed[face, order] for face, order in mappings])


def vertex_chirality(vertices):
    """Four oriented crease-ray volumes at each of 16 interior vertices."""
    signatures = []
    for row in range(1, SIZE):
        for column in range(1, SIZE):
            rays = np.array([vertices[row - 1, column], vertices[row, column + 1],
                             vertices[row + 1, column], vertices[row, column - 1]]) - vertices[row, column]
            signatures.append([np.linalg.det(rays[[(offset + step) % 4 for step in range(3)]])
                               for offset in range(4)])
    return np.array(signatures)


def cross2(first, second):
    return first[0] * second[1] - first[1] * second[0]


def coplanar_overlap(first, second, normal, tolerance=1e-9):
    """Positive overlap area of two coplanar convex quadrilaterals."""
    keep = np.arange(3) != np.argmax(np.abs(normal))
    subject = list(first[:, keep])
    clip = second[:, keep]
    orientation = np.sign(sum(cross2(point, clip[(index + 1) % 4]) for index, point in enumerate(clip)))
    for index, start in enumerate(clip):
        direction = clip[(index + 1) % 4] - start
        if not subject:
            return False
        result = []
        previous = subject[-1]
        previous_distance = orientation * cross2(direction, previous - start)
        for current in subject:
            current_distance = orientation * cross2(direction, current - start)
            if (current_distance >= 0) != (previous_distance >= 0):
                fraction = previous_distance / (previous_distance - current_distance)
                result.append(previous + fraction * (current - previous))
            if current_distance >= 0:
                result.append(current)
            previous, previous_distance = current, current_distance
        subject = result
    if len(subject) < 3:
        return False
    area = abs(sum(cross2(point, subject[(index + 1) % len(subject)])
                   for index, point in enumerate(subject))) / 2
    return area > tolerance


def check_intersections(faces):
    """Check sampled transverse crossings and coplanar area overlaps."""
    lower, upper = faces.min(axis=1), faces.max(axis=1)
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 3] - faces[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    for first in range(len(faces)):
        for second in range(first + 1, len(faces)):
            if np.any(upper[first] < lower[second] - 1e-9) or np.any(upper[second] < lower[first] - 1e-9):
                continue
            if abs(np.dot(normals[first], normals[second])) > 1 - 1e-10:
                if np.max(np.abs((faces[second] - faces[first, 0]) @ normals[first])) < 1e-9:
                    if coplanar_overlap(faces[first], faces[second], normals[first]):
                        raise AssertionError(f'Coplanar overlap between panels {first + 1} and {second + 1}.')
                continue
            if (base_model.pierces_panel(faces[first], faces[second])
                    or base_model.pierces_panel(faces[second], faces[first])):
                raise AssertionError(f'Transverse intersection between panels {first + 1} and {second + 1}.')


def validate(path, require_isotropic=False):
    """Validate the full mesh, not just its seed or outer boundary."""
    if len(SHARED_EDGES) != 40 or RHOMBI.sum() != 13:
        raise AssertionError('Expected 13 rhombi, 12 squares and 40 shared edges.')
    pattern = RHOMBI.reshape(SIZE, SIZE)
    counts = []
    for row in range(3):
        for column in range(3):
            window = pattern[row:row + 3, column:column + 3]
            expected = (np.indices((3, 3)).sum(axis=0) + row + column) % 2 == 0
            np.testing.assert_array_equal(window, expected)
            counts.append(int(window.sum()))
    if sorted(counts) != [4] * 4 + [5] * 5:
        raise AssertionError('Unexpected 3x3 window patterns.')
    initial_vertices, initial_faces = grid_configuration(path[0])
    reference_distances = np.linalg.norm(initial_faces[:, :, None] - initial_faces[:, None, :], axis=-1)
    initial_signs = np.sign(vertex_chirality(initial_vertices))
    worst = dict(rigidity=0.0, side_length=0.0, angle=0.0, planarity=0.0,
                 shared_edge=0.0, central_3x3=0.0)
    chirality_margin = np.inf
    for rotations in path:
        vertices, faces = grid_configuration(rotations)
        distances = np.linalg.norm(faces[:, :, None] - faces[:, None, :], axis=-1)
        worst['rigidity'] = max(worst['rigidity'], np.max(np.abs(distances - reference_distances)))
        old = faces.reshape(SIZE, SIZE, 4, 3)[1:4, 1:4].reshape(9, 4, 3)
        worst['central_3x3'] = max(worst['central_3x3'], np.max(np.abs(old - central_reference(rotations))))
        for index, face in enumerate(faces):
            lengths, angles = base_model.face_stats(face)
            expected = np.array([60, 120, 60, 120]) if RHOMBI[index] else np.full(4, 90)
            worst['side_length'] = max(worst['side_length'], np.max(np.abs(lengths - 1)))
            worst['angle'] = max(worst['angle'], np.max(np.abs(angles - expected)))
            normal = base_model.norm(np.cross(face[1] - face[0], face[3] - face[0]))
            worst['planarity'] = max(worst['planarity'], abs(np.dot(face[2] - face[0], normal)))
        for (first, first_pair), (second, second_pair) in SHARED_EDGES:
            gap = np.max(np.linalg.norm(faces[first, first_pair] - faces[second, second_pair], axis=1))
            worst['shared_edge'] = max(worst['shared_edge'], gap)
        chirality = vertex_chirality(vertices)
        if not np.array_equal(np.sign(chirality), initial_signs):
            raise AssertionError(f'An interior vertex inverted at drive angle {rotations[0]:.6f}.')
        chirality_margin = min(chirality_margin, np.min(np.abs(chirality)))
        check_intersections(faces)
    for name, error in worst.items():
        print(f'max {name} error = {error:.3e}')
        if error > 1e-8:
            raise AssertionError(f'{name} error exceeds tolerance: {error}')
    if chirality_margin < 1e-8:
        raise AssertionError('An interior vertex approaches a chirality degeneracy.')
    print(f'min chirality margin = {chirality_margin:.6f}; all 16 interior vertices keep their signs.')
    print('All 9 overlapping 3x3 windows preserve the alternating pattern; the central window matches the original.')
    print(f'PASS: {len(path)} states; 25 rigid planar faces; 40 shared edges; no sampled panel intersections.')
    base_model.contraction_report([base_model.row_column_spans(grid_configuration(angles)[0])
                                   for angles in path], require_isotropic)
    return worst


def prepare_axis(axis, points):
    points = np.asarray(points).reshape(-1, 3)
    lower, upper = points.min(axis=0), points.max(axis=0)
    axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
    axis.set_box_aspect(upper - lower, zoom=1)
    axis.set_proj_type('ortho')
    axis.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH, roll=VIEW_ROLL)
    axis.set_axis_off()
    axis.set_facecolor('#f5f7fb')
    return upper - lower


def fit_axis(axis, points, span):
    """Fit all motion samples with one constant, physically uniform scale."""
    horizontal, vertical, _ = proj3d.proj_transform(*np.asarray(points).reshape(-1, 3).T, axis.get_proj())
    pixels = axis.transData.transform(np.column_stack([horizontal, vertical]))
    origin = axis.transData.transform([0, 0])
    margin = 0.065 * min(axis.bbox.width, axis.bbox.height)
    available = np.array([origin[0] - axis.bbox.x0 - margin, axis.bbox.x1 - origin[0] - margin,
                          origin[1] - axis.bbox.y0 - margin, axis.bbox.y1 - origin[1] - margin])
    extent = np.array([origin[0] - pixels[:, 0].min(), pixels[:, 0].max() - origin[0],
                       origin[1] - pixels[:, 1].min(), pixels[:, 1].max() - origin[1]])
    axis.set_box_aspect(span, zoom=float(np.min(available / extent)))


def panel_mesh(faces):
    return Poly3DCollection(faces, facecolors=COLORS, edgecolors='#203349', linewidths=1.1)


def render_static(faces, output, angle, title, show_labels=True):
    figure = plt.figure(figsize=(9, 8.5), facecolor='#f5f7fb')
    axis = figure.add_axes([0.04, 0.14, 0.92, 0.73], projection='3d')
    axis.add_collection3d(panel_mesh(faces))
    span = prepare_axis(axis, faces)
    if show_labels:
        for index, face in enumerate(faces):
            row, column = divmod(index, SIZE)
            axis.text(*face.mean(axis=0), f'{row + 1},{column + 1}', fontsize=8,
                      ha='center', va='center', color='#233e53', zorder=10,
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    figure.suptitle('5 × 5 Rhombi + Squares', fontsize=20, fontweight='bold', y=0.96)
    figure.text(0.5, 0.9, f'{title} · Drive angle: {angle:.2f}°', ha='center', fontsize=12, color='#46566a')
    figure.text(0.5, 0.095, '13 blue rhombi · 12 orange squares · 40 shared edges', ha='center', fontsize=11, color='#46566a')
    caption = 'Labels: (row, column)' if show_labels else '16 interior vertices keep their chirality'
    figure.text(0.5, 0.052, caption + ' · Central 3 × 3 matches the original model', ha='center', fontsize=10, color='#46566a')
    figure.canvas.draw()
    fit_axis(axis, faces, span)
    figure.savefig(output, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def render(path, output, fps=25):
    configurations = np.array([grid_configuration(rotations)[1] for rotations in path])
    assembled = output.with_name(output.stem + '_assembled.png')
    folded = output.with_name(output.stem + '_folded.png')
    render_static(configurations[0], assembled, path[0, 0], 'Initial connected assembly')
    render_static(configurations[-1], folded, path[-1, 0], 'Folded configuration', show_labels=False)
    figure = plt.figure(figsize=(14, 7.6), facecolor='#f5f7fb')
    layout = figure.add_gridspec(1, 2, wspace=0.03)
    fixed_axis = figure.add_subplot(layout[0], projection='3d')
    moving_axis = figure.add_subplot(layout[1], projection='3d')
    fixed_axis.add_collection3d(panel_mesh(configurations[0]))
    moving_mesh = panel_mesh(configurations[0])
    moving_axis.add_collection3d(moving_mesh)
    spans = [prepare_axis(axis, configurations) for axis in (fixed_axis, moving_axis)]
    fixed_axis.set_title('Initial assembly · fixed', fontsize=13, pad=10)
    moving_axis.set_title('Fold / return', fontsize=13, pad=10)
    figure.suptitle('5 × 5 Rigid Folding · Rhombi + Squares', fontsize=19, fontweight='bold', y=0.96)
    figure.text(0.27, 0.14, '25 connected planar faces\n13 rhombi + 12 squares', ha='center', fontsize=11, color='#46566a', linespacing=1.5)
    state = figure.text(0.755, 0.14, '', ha='center', fontsize=11, color='#46566a', linespacing=1.5)
    figure.text(0.5, 0.05, '40 shared edges · 16 vertex signs fixed · Same fixed scale in both views', ha='center', fontsize=10, color='#46566a')
    figure.subplots_adjust(left=0.025, right=0.985, top=0.84, bottom=0.23)
    figure.canvas.draw()
    fit_axis(fixed_axis, configurations, spans[0])
    fit_axis(moving_axis, configurations, spans[1])
    hold = max(1, fps // 2)
    indices = np.concatenate([np.repeat(0, hold), np.arange(1, len(path)),
                              np.repeat(len(path) - 1, hold), np.arange(len(path) - 2, 0, -1)])

    def update(frame):
        index = indices[frame]
        faces = configurations[index]
        moving_mesh.set_verts(faces)
        state.set_text(f'Drive angle: {path[index, 0]:.1f}°\nHeight span: {np.ptp(faces[:, :, 2]):.3f}')
        return moving_mesh, state

    update(0)
    animation = FuncAnimation(figure, update, frames=len(indices), interval=1000 / fps, blit=False)
    animation.save(str(output), writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    print(f'Initial assembly: {assembled.resolve()}\nFolded state: {folded.resolve()}\nGIF: {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-angle', type=float, default=base_model.START_ANGLE)
    parser.add_argument('--end-angle', type=float, default=base_model.END_ANGLE)
    parser.add_argument('--steps', type=int, default=80, help='Steps per half cycle (>=2).')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--require-isotropic', action='store_true',
                        help='Reject paths without equal fractional H/V contraction before writing images.')
    args = parser.parse_args()
    if args.fps <= 0 or args.output.suffix.lower() != '.gif':
        parser.error('Need fps > 0 and an output filename ending in .gif.')
    path = base_model.fold_path(args.steps, args.start_angle, args.end_angle)
    try:
        validate(path, args.require_isotropic)
    except ValueError as error:
        parser.error(str(error))
    if not args.check_only:
        render(path, args.output, args.fps)


if __name__ == '__main__':
    main()
