"""Exact five-strip accordion with 25 rigid nonplanar quadrilaterals."""

import argparse
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgb
import numpy as np
from PIL import Image

from build_geogebra_screw_folding5x5 import FACE_IDS, TRIANGLE_IDS, add_command, edge_incidence, hinge_angles, vertex_signatures


PROJECT = Path(__file__).resolve().parent
STEM = '5x5 nonplanar accordion folding'
WIDTH = 2.0
PITCH = 0.8
TWIST = 12.0
MIN_FOLD = 30.0
MAX_FOLD = 150.0
COLORS = ['#66a5cf' if (row + column) % 2 == 0 else '#f0ca73' for row in range(5) for column in range(5)]
EDGES = np.array(list(edge_incidence()))
ACTIVE_EDGES = np.array([first // 6 == second // 6 and 1 <= first // 6 <= 4 for first, second in EDGES])


def base_geometry():
    row, column = np.indices((6, 6))
    skew = (-1.0)**row * np.radians(TWIST / 2)
    return np.stack(((row - 2) * WIDTH, (column - 2.5) * PITCH * np.cos(skew),
                     (column - 2.5) * PITCH * np.sin(skew)), axis=-1)


def rotate(points, origin, direction, angle):
    direction = direction / np.linalg.norm(direction)
    offsets = points - origin
    return (origin + offsets * np.cos(angle) + np.cross(direction, offsets) * np.sin(angle)
            + np.outer(offsets @ direction, direction) * (1 - np.cos(angle)))


def geometry(degrees):
    positions = base_geometry()
    for boundary in (2, 1):
        positions[:boundary] = rotate(positions[:boundary].reshape(-1, 3), positions[boundary, 0],
                                      positions[boundary, -1] - positions[boundary, 0],
                                      -(-1.0)**boundary * np.radians(degrees)).reshape(boundary, 6, 3)
    for boundary in (3, 4):
        positions[boundary + 1:] = rotate(positions[boundary + 1:].reshape(-1, 3), positions[boundary, 0],
                                          positions[boundary, -1] - positions[boundary, 0],
                                          (-1.0)**boundary * np.radians(degrees)).reshape(5 - boundary, 6, 3)
    return positions


def analytic_signatures(degrees):
    row, column = np.indices((4, 4)) + 1
    transverse = (column - 2.5) * PITCH
    sine_twist = np.sin(np.radians(TWIST))
    angle = np.radians(degrees)
    determinant = (-1.0)**row * PITCH * (
        (WIDTH**2 - transverse**2 * sine_twist**2) * np.sin(angle)
        + 2 * WIDTH * transverse * sine_twist * np.cos(angle))
    zeros = np.zeros_like(determinant)
    return np.stack((determinant, zeros, determinant, zeros), axis=-1).reshape(16, 4)


def validate(samples=1201):
    initial, final = geometry(MIN_FOLD), geometry(MAX_FOLD)
    panels = initial.reshape(-1, 3)[FACE_IDS]
    reference = np.linalg.norm(panels[:, :, None] - panels[:, None, :], axis=-1)
    volumes = np.linalg.det(panels[:, 1:] - panels[:, :1])
    errors = {'pair_distance': 0., 'signed_six_volume': 0., 'fixed_strip': 0., 'signature_identity': 0.}
    initial_signs = vertex_signatures(initial)
    active_signs = np.abs(initial_signs) > 1e-9
    minimum_signature = np.inf
    for degree in np.linspace(MIN_FOLD, MAX_FOLD, samples):
        points = geometry(degree)
        faces = points.reshape(-1, 3)[FACE_IDS]
        distances = np.linalg.norm(faces[:, :, None] - faces[:, None, :], axis=-1)
        errors['pair_distance'] = max(errors['pair_distance'], float(np.max(np.abs(distances - reference))))
        errors['signed_six_volume'] = max(errors['signed_six_volume'], float(np.max(np.abs(
            np.linalg.det(faces[:, 1:] - faces[:, :1]) - volumes))))
        errors['fixed_strip'] = max(errors['fixed_strip'], float(np.max(np.abs(points[2:4] - initial[2:4]))))
        signatures = vertex_signatures(points)
        errors['signature_identity'] = max(errors['signature_identity'], float(np.max(np.abs(
            signatures - analytic_signatures(degree)))))
        assert np.array_equal(np.sign(signatures[active_signs]), np.sign(initial_signs[active_signs]))
        minimum_signature = min(minimum_signature, float(np.min(np.abs(signatures[active_signs]))))
    classes = []
    for index, distances in enumerate(reference):
        permutations = ([0, 1, 2, 3], [2, 3, 0, 1], [0, 3, 2, 1], [2, 1, 0, 3])
        candidates = [distances[np.ix_(permutation, permutation)] for permutation in permutations]
        for representative, members in classes:
            if any(np.max(np.abs(representative - candidate)) < 1e-10 for candidate in candidates):
                members.append(index + 1)
                break
        else:
            classes.append((distances, [index + 1]))
    hinge_change = np.degrees(np.abs(np.angle(np.exp(1j * (hinge_angles(final) - hinge_angles(initial))))))
    assert np.count_nonzero(hinge_change > 1e-8) == 20
    assert np.max(np.abs(hinge_change[hinge_change > 1e-8] - 120)) < 1e-8
    assert len(classes) == 3 and max(errors.values()) < 1e-10
    assert np.max(np.abs(np.abs(volumes) - WIDTH * PITCH**2 * np.sin(np.radians(TWIST)))) < 1e-10
    start_span, end_span = np.ptp(initial.reshape(-1, 3), axis=0), np.ptp(final.reshape(-1, 3), axis=0)
    return {
        'model': 'Five rigid skew strips, four alternating straight hinges, 25 nonplanar cells',
        'faces': 25, 'vertices': 36, 'whole_edge_connections': 40, 'moving_connections': 20,
        'fixed_angle_connections': 20, 'straight_hinge_lines': 4,
        'congruence_classes_including_reflections': [members for _, members in classes],
        'all_faces_congruent': False, 'all_faces_nonplanar': True,
        'rigid_by_exact_rotation_construction': True, 'internal_diagonals_are_not_hinges': True,
        'driver_interval_degrees': [MIN_FOLD, MAX_FOLD], 'strip_pitch': WIDTH,
        'column_pitch': PITCH, 'base_boundary_twist_degrees': TWIST,
        'sampled_configurations': samples, 'maximum_errors': errors,
        'absolute_face_signed_six_volume': float(abs(volumes[0])),
        'nonzero_vertex_signatures': int(active_signs.sum()), 'identically_zero_signatures': 32,
        'vertex_signature_nonzero_signs_preserved_by_analytic_bound': True,
        'minimum_sampled_absolute_nonzero_signature': minimum_signature,
        'initial_xyz_span': start_span.tolist(), 'final_xyz_span': end_span.tolist(),
        'x_span_contraction_percent': float(100 * (1 - end_span[0] / start_span[0])),
        'scope': 'New geometry, not the original model. 20 of 40 shared edges move; 20 are fixed. Not flat folding.'
    }


def build_ggb(output, thumbnail):
    with ZipFile(PROJECT / 'geogebra/2x2 nonplanar folding.ggb') as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(contents['geogebra.xml'])
    construction = root.find('construction')
    templates = {element.get('label'): deepcopy(element) for element in construction.findall('element')}
    construction.clear()
    construction.set('title', '5x5 nonplanar accordion folding - 20 active and 20 fixed shared edges')
    initial, base = geometry(MIN_FOLD), base_geometry()

    def numeric(label, expression, value, slider=False):
        if expression is not None:
            ET.SubElement(construction, 'expression', label=label, exp=expression, type='numeric')
        element = ET.SubElement(construction, 'element', type='numeric', label=label)
        ET.SubElement(element, 'value', val=str(value))
        ET.SubElement(element, 'show', object=str(slider).lower(), label='true')
        ET.SubElement(element, 'objColor', r='189', g='66', b='67', alpha='0')
        ET.SubElement(element, 'labelMode', val='0')
        ET.SubElement(element, 'auxiliary', val='false')
        if slider:
            ET.SubElement(element, 'slider', min=str(MIN_FOLD), max=str(MAX_FOLD), absoluteScreenLocation='true',
                          width='220', x='35', y='55', fixed='false', horizontal='true', showAlgebra='true')
            ET.SubElement(element, 'lineStyle', thickness='4', type='0', typeHidden='1')
            ET.SubElement(element, 'animation', step='0.1', speed='0.7', type='0', playing='false')
            ET.SubElement(element, 'caption', val='手风琴折叠参数（度）')

    def point(label, coordinates, expression=None, command=None):
        if expression is not None:
            ET.SubElement(construction, 'expression', label=label, exp=expression, type='point')
        if command is not None:
            add_command(construction, command[0], command[1], [label])
        element = deepcopy(templates['A'])
        element.set('label', label)
        element.find('coords').attrib.update(dict(zip(('x', 'y', 'z'), map(str, coordinates))), w='1')
        element.find('show').attrib.update(object='false', label='false')
        ET.SubElement(element, 'auxiliary', val='false')
        construction.append(element)

    def axis(row):
        label = f'axis_{{{row}}}'
        add_command(construction, 'Line', [f'P_{{{row}0}}', f'P_{{{row}5}}'], [label])
        element = ET.SubElement(construction, 'element', type='line3d', label=label)
        ET.SubElement(element, 'show', object='false', label='false')
        ET.SubElement(element, 'objColor', r='189', g='66', b='67', alpha='0')
        ET.SubElement(element, 'lineStyle', thickness='3', type='0', typeHidden='1')
        ET.SubElement(element, 'auxiliary', val='false')
        return label

    numeric('fold', None, MIN_FOLD, slider=True)
    numeric('theta', 'fold*pi/180', np.radians(MIN_FOLD))
    numeric('stripWidth', '2', WIDTH)
    numeric('pitch', '4/5', PITCH)
    numeric('beta', 'pi/30', np.radians(TWIST / 2))
    for row in range(6):
        for column in range(6):
            expression = (f'(({row}-2)*stripWidth,({column}-5/2)*pitch*cos(beta),'
                          f'({column}-5/2)*pitch*sin(beta)*({(-1)**row}))')
            point(f'B_{{{row}{column}}}', base[row, column], expression=expression)
    for row in (2, 3):
        for column in range(6):
            point(f'P_{{{row}{column}}}', initial[row, column], expression=f'B_{{{row}{column}}}')
    for hinge, inner, outer in ((2, 1, 0), (3, 4, 5)):
        first_axis = axis(hinge)
        intermediate = rotate(base[outer], base[hinge, 0], base[hinge, 5] - base[hinge, 0], -np.radians(MIN_FOLD))
        for column in range(6):
            point(f'P_{{{inner}{column}}}', initial[inner, column],
                  command=('Rotate', [f'B_{{{inner}{column}}}', '-theta', first_axis]))
            point(f'Q_{{{outer}{column}}}', intermediate[column],
                  command=('Rotate', [f'B_{{{outer}{column}}}', '-theta', first_axis]))
        second_axis = axis(inner)
        for column in range(6):
            point(f'P_{{{outer}{column}}}', initial[outer, column],
                  command=('Rotate', [f'Q_{{{outer}{column}}}', 'theta', second_axis]))
    labels = [f'P_{{{row}{column}}}' for row in range(6) for column in range(6)]
    for face_index, corners in enumerate(FACE_IDS):
        row, column = divmod(face_index, 5)
        for triangle_index, indices in enumerate(((0, 1, 2), (0, 2, 3))):
            label = f'panel_{{{row+1}{column+1}{triangle_index}}}'
            inner_edges = [f'inner_{{{row+1}{column+1}{triangle_index}{edge}}}' for edge in range(3)]
            add_command(construction, 'Polygon', [labels[corners[index]] for index in indices], [label, *inner_edges])
            face = deepcopy(templates['t5'])
            face.set('label', label)
            face.find('lineStyle').set('thickness', '0')
            face.find('show').attrib.update(object='true', label='false')
            color = [round(value * 255) for value in to_rgb(COLORS[face_index])]
            face.find('objColor').attrib.update(dict(zip(('r', 'g', 'b'), map(str, color))), alpha='1')
            ET.SubElement(face, 'auxiliary', val='false')
            construction.append(face)
            for edge_label in inner_edges:
                edge = deepcopy(templates['f'])
                edge.set('label', edge_label)
                edge.find('show').attrib.update(object='false', label='false')
                ET.SubElement(edge, 'auxiliary', val='false')
                construction.append(edge)
    for index, endpoints in enumerate(EDGES):
        active = bool(ACTIVE_EDGES[index])
        label = f'{"hinge" if active else "edge"}_{{{index+1}}}'
        add_command(construction, 'Segment', [labels[endpoint] for endpoint in endpoints], [label])
        edge = deepcopy(templates['f'])
        edge.set('label', label)
        edge.find('show').attrib.update(object='true', label='false')
        edge.find('objColor').attrib.update(dict(zip(('r', 'g', 'b'), ('189', '66', '67') if active else ('35', '60', '77'))))
        edge.find('lineStyle').set('thickness', '5' if active else '2')
        ET.SubElement(edge, 'auxiliary', val='false')
        construction.append(edge)
    for path, value in (('kernel/angleUnit', 'radian'), ('kernel/decimals', '6'),
                         ('kernel/algebraStyle', '3'), ('algebraView/mode', '3')):
        root.find(path).set('val', value)
    root.find('euclidianView3D/coordSystem').attrib.update(
        xZero='-1', yZero='0', zZero='0', scale='65', xAngle='40', zAngle='-26')
    for view in root.findall('gui/perspectives/perspective/views/view'):
        view.set('visible', str(view.get('id') in ('2', '512')).lower())
    contents['geogebra.xml'] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    contents['geogebra_thumbnail.png'] = thumbnail
    contents['geogebra_javascript.js'] = b'function ggbOnInit() {}'
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)


def projection_basis():
    elevation, azimuth = np.radians([40., -64.])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
    toward = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    return np.array([right, np.cross(toward, right), toward])


def raster_surface(vertices, center, scale, size=(1800, 1200)):
    image_width, image_height = size
    projected = vertices.reshape(-1, 3) @ projection_basis().T
    projected[:, :2] = (projected[:, :2] - center) * scale
    projected[:, 0] += image_width / 2
    projected[:, 1] = image_height / 2 - projected[:, 1]
    canvas = np.full((image_height, image_width, 3), [246, 248, 251], dtype=np.uint8)
    depth = np.full((image_height, image_width), -np.inf)
    triangles = vertices.reshape(-1, 3)[TRIANGLE_IDS]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    light = np.array([.4, -.5, .85])
    light /= np.linalg.norm(light)
    shade = (.72 + .28 * np.abs(normals @ light))[:, None]
    colors = np.rint(255 * np.repeat(np.array([to_rgb(color) for color in COLORS]), 2, axis=0) * shade).astype(np.uint8)
    for triangle, color in zip(projected[TRIANGLE_IDS], colors):
        lower = np.maximum(np.floor(triangle[:, :2].min(axis=0)).astype(int), [0, 0])
        upper = np.minimum(np.ceil(triangle[:, :2].max(axis=0)).astype(int), [image_width - 1, image_height - 1])
        if np.any(lower > upper):
            continue
        horizontal, vertical = np.meshgrid(np.arange(lower[0], upper[0] + 1), np.arange(lower[1], upper[1] + 1))
        first, second, third = triangle
        denominator = (second[1] - third[1]) * (first[0] - third[0]) + (third[0] - second[0]) * (first[1] - third[1])
        if abs(denominator) < 1e-10:
            continue
        first_weight = ((second[1] - third[1]) * (horizontal - third[0]) + (third[0] - second[0]) * (vertical - third[1])) / denominator
        second_weight = ((third[1] - first[1]) * (horizontal - third[0]) + (first[0] - third[0]) * (vertical - third[1])) / denominator
        third_weight = 1 - first_weight - second_weight
        candidate_depth = first_weight * first[2] + second_weight * second[2] + third_weight * third[2]
        visible = ((first_weight >= -1e-9) & (second_weight >= -1e-9) & (third_weight >= -1e-9)
                   & (candidate_depth > depth[vertical, horizontal]))
        depth[vertical[visible], horizontal[visible]] = candidate_depth[visible]
        canvas[vertical[visible], horizontal[visible]] = color
    for endpoints, active in zip(EDGES, ACTIVE_EDGES):
        first, second = projected[endpoints]
        thickness = 4.0 if active else 2.0
        lower = np.maximum(np.floor(np.minimum(first[:2], second[:2]) - thickness).astype(int), [0, 0])
        upper = np.minimum(np.ceil(np.maximum(first[:2], second[:2]) + thickness).astype(int), [image_width - 1, image_height - 1])
        if np.any(lower > upper):
            continue
        horizontal, vertical = np.meshgrid(np.arange(lower[0], upper[0] + 1), np.arange(lower[1], upper[1] + 1))
        direction = second[:2] - first[:2]
        denominator = np.dot(direction, direction)
        if denominator < 1e-10:
            continue
        parameter = np.clip(((horizontal - first[0]) * direction[0] + (vertical - first[1]) * direction[1]) / denominator, 0, 1)
        distance_squared = ((horizontal - first[0] - parameter * direction[0])**2
                            + (vertical - first[1] - parameter * direction[1])**2)
        candidate_depth = first[2] + parameter * (second[2] - first[2])
        visible = (distance_squared <= thickness**2) & (candidate_depth >= depth[vertical, horizontal] - .02)
        canvas[vertical[visible], horizontal[visible]] = [189, 66, 67] if active else [35, 60, 77]
    return canvas, projected


def render(static_path, animation_path, frames=101):
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    degrees = MIN_FOLD + (MAX_FOLD - MIN_FOLD) * (1 - np.cos(np.linspace(0, 2 * np.pi, frames))) / 2
    configurations = [geometry(value) for value in degrees]
    projected = np.array(configurations).reshape(-1, 3) @ projection_basis().T
    lower, upper = projected[:, :2].min(axis=0), projected[:, :2].max(axis=0)
    center = (lower + upper) / 2
    scale = min(1620 / (upper[0] - lower[0]), 1060 / (upper[1] - lower[1]))
    initial = geometry(MIN_FOLD)
    initial_width = np.ptp(initial[:, :, 0])
    initial_image, initial_projected = raster_surface(initial, center, scale)
    figure = plt.figure(figsize=(12, 8), facecolor='#f6f8fb')
    axis = figure.add_axes([.01, .1, .98, .76])
    axis.imshow(initial_image)
    axis.set_axis_off()
    for index, panel in enumerate(initial_projected[FACE_IDS]):
        label_position = panel.mean(axis=0)
        axis.text(*label_position[:2], f'{index//5+1},{index%5+1}', ha='center', va='center', fontsize=8,
                  color='#20394b', bbox={'fc': 'white', 'ec': 'none', 'alpha': .6, 'pad': 1})
    figure.suptitle('5 × 5 空间四边形 · 手风琴式刚性折叠', fontsize=20, y=.97)
    figure.text(.5, .9, '初始拼接 · 25 个非平面面片，3 类形状（镜像归同类）', ha='center', fontsize=11, color='#4c6376')
    figure.text(.5, .055, '红线：20 段可动共边 / 4 条折痕；其余 20 段内部共边保持固定夹角', ha='center', fontsize=11, color='#4c6376')
    figure.savefig(static_path, dpi=140)
    buffer = BytesIO()
    figure.savefig(buffer, format='png', dpi=35)
    plt.close(figure)
    thumbnail = Image.open(buffer)
    thumbnail.thumbnail((256, 256))
    packed = BytesIO()
    thumbnail.save(packed, format='PNG')
    figure = plt.figure(figsize=(14, 7), facecolor='#f6f8fb')
    fixed_axis = figure.add_axes([.01, .15, .48, .68])
    moving_axis = figure.add_axes([.51, .15, .48, .68])
    fixed_axis.imshow(initial_image)
    moving_image = moving_axis.imshow(initial_image)
    fixed_axis.set_axis_off()
    moving_axis.set_axis_off()
    figure.suptitle('5 × 5 空间四边形 · 交替收拢，不拉伸面片', fontsize=20, y=.97)
    figure.text(.25, .88, '初始拼接 · 相机与比例固定', ha='center', fontsize=12, color='#34556f')
    caption = figure.text(.75, .88, '', ha='center', fontsize=12, color='#34556f')
    figure.text(.5, .095, '25 面均刚性且非平面 · 40 条共边始终相连 · 中央刚性带固定', ha='center', fontsize=12, color='#34556f')
    figure.text(.5, .045, '20 段共边参与折叠（红色），另外 20 段夹角固定；不是 40 段全部活动', ha='center', fontsize=11, color='#526e83')

    def update(index):
        moving_image.set_data(raster_surface(configurations[index], center, scale)[0])
        contraction = 100 * (1 - np.ptp(configurations[index][:, :, 0]) / initial_width)
        caption.set_text(f'参数 {degrees[index]:.1f}° · 横向缩短 {contraction:.1f}%')
        return []

    animation = FuncAnimation(figure, update, frames=frames, interval=65, blit=False)
    animation.save(animation_path, writer=PillowWriter(fps=18), dpi=100)
    plt.close(figure)
    return packed.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--frames', type=int, default=101)
    arguments = parser.parse_args()
    report = validate()
    if not arguments.check_only:
        thumbnail = render(PROJECT / 'previews' / (STEM + '.png'), PROJECT / 'animations' / (STEM + '.gif'), arguments.frames)
        build_ggb(PROJECT / 'geogebra' / (STEM + '.ggb'), thumbnail)
        (PROJECT / 'geogebra' / (STEM + '.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
