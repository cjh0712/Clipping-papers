"""Exact congruent skew-quadrilateral screw net with a continuous rigid flex."""

import argparse
from collections import Counter
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image


PROJECT = Path(__file__).resolve().parent
STEM = '5x5 nonplanar rigid screw folding'
HALF_LONG = 2.0
HALF_SHORT = 1.0
OFFSET = 0.2
MAX_DEGREES = 45.0
GRID_IDS = np.arange(36).reshape(6, 6)
FACE_IDS = np.array([[GRID_IDS[row, column], GRID_IDS[row, column + 1],
                      GRID_IDS[row + 1, column + 1], GRID_IDS[row + 1, column]]
                     for row in range(5) for column in range(5)])
TRIANGLE_IDS = np.array([face[list(corners)] for face in FACE_IDS for corners in ((0, 1, 2), (0, 2, 3))])
COLORS = ['#569bcd' if (row + column) % 2 == 0 else '#edc55f' for row in range(5) for column in range(5)]
MASTER = np.array([[0., -HALF_LONG, 0.], [OFFSET, 0., -HALF_SHORT],
                   [0., HALF_LONG, 0.], [OFFSET, 0., HALF_SHORT]])


def geometry(degrees):
    angle = np.radians(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    transverse_long, transverse_short = HALF_LONG * cosine, -HALF_SHORT * sine
    shift = (transverse_long**2 - transverse_short**2 - OFFSET**2) / (2 * OFFSET)
    radius = np.hypot(shift, transverse_long)
    half_sum = np.arctan2(transverse_long, shift)
    half_difference = np.arctan2(transverse_short, shift + OFFSET)
    row_turn, column_turn = half_sum + half_difference, half_sum - half_difference
    row_rise = HALF_LONG * sine + HALF_SHORT * cosine
    column_rise = HALF_LONG * sine - HALF_SHORT * cosine
    row_indices, column_indices = np.indices((6, 6)) - 2
    phase = row_indices * row_turn + column_indices * column_turn - half_sum
    height = row_indices * row_rise + column_indices * column_rise - HALF_LONG * sine
    vertices = np.stack((-shift + radius * np.cos(phase),
                         radius * np.sin(phase) * cosine + height * sine,
                         -radius * np.sin(phase) * sine + height * cosine), axis=-1)
    scalars = {'theta': angle, 'transverseLong': transverse_long, 'transverseShort': transverse_short,
               'axisShift': shift, 'radius': radius, 'halfSum': half_sum, 'halfDifference': half_difference,
               'rowTurn': row_turn, 'columnTurn': column_turn, 'rowRise': row_rise, 'columnRise': column_rise}
    return vertices, scalars


def vertex_signatures(vertices):
    signatures = []
    for row in range(1, 5):
        for column in range(1, 5):
            rays = vertices[[row - 1, row, row + 1, row], [column, column + 1, column, column - 1]] - vertices[row, column]
            signatures.append([np.linalg.det(rays[[(start + step) % 4 for step in range(3)]]) for start in range(4)])
    return np.array(signatures)


def edge_incidence():
    incidence = {}
    for face_index, face in enumerate(FACE_IDS):
        for corner in range(4):
            first, second = int(face[corner]), int(face[(corner + 1) % 4])
            incidence.setdefault(tuple(sorted((first, second))), []).append((face_index, corner))
    return incidence


def hinge_angles(vertices):
    flat = vertices.reshape(-1, 3)
    triangles = flat[TRIANGLE_IDS]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    result = []
    for (first, second), adjacent in edge_incidence().items():
        if len(adjacent) != 2:
            continue
        first_face, first_corner = adjacent[0]
        second_face, second_corner = adjacent[1]
        first_normal = normals[2 * first_face + int(first_corner >= 2)]
        second_normal = normals[2 * second_face + int(second_corner >= 2)]
        direction = flat[second] - flat[first]
        direction /= np.linalg.norm(direction)
        result.append(np.arctan2(np.dot(direction, np.cross(first_normal, second_normal)),
                                 np.dot(first_normal, second_normal)))
    return np.array(result)


def validate(samples=901):
    initial, _ = geometry(0)
    expected_distances = np.linalg.norm(MASTER[:, None] - MASTER[None, :], axis=-1)
    expected_volume = np.linalg.det(MASTER[1:] - MASTER[0])
    initial_signs = np.sign(vertex_signatures(initial))
    max_distance_error, max_anchor_error, max_volume_error = 0., 0., 0.
    minimum_signature = np.inf
    for degrees in np.linspace(0, MAX_DEGREES, samples):
        vertices, _ = geometry(degrees)
        panels = vertices.reshape(-1, 3)[FACE_IDS]
        distances = np.linalg.norm(panels[:, :, None] - panels[:, None, :], axis=-1)
        max_distance_error = max(max_distance_error, float(np.max(np.abs(distances - expected_distances))))
        max_anchor_error = max(max_anchor_error, float(np.max(np.abs(panels[12] - MASTER))))
        volumes = np.linalg.det(panels[:, 1:] - panels[:, :1])
        max_volume_error = max(max_volume_error, float(np.max(np.abs(volumes - expected_volume))))
        signatures = vertex_signatures(vertices)
        assert np.array_equal(np.sign(signatures), initial_signs)
        minimum_signature = min(minimum_signature, float(np.min(np.abs(signatures))))
    counts = Counter(len(value) for value in edge_incidence().values())
    assert counts == {1: 20, 2: 40}
    assert max(max_distance_error, max_anchor_error, max_volume_error) < 1e-10
    assert abs(expected_volume) > 1 and minimum_signature > .5
    final, _ = geometry(MAX_DEGREES)
    hinge_change = np.degrees(np.abs(np.angle(np.exp(1j * (hinge_angles(final) - hinge_angles(initial))))))
    assert np.min(hinge_change) > 1
    report = {
        'model': 'Exact congruent skew rhombi on a variable cylinder; commuting screw motions',
        'face_count': 25, 'vertices': 36, 'whole_edge_hinges': 40,
        'all_faces_congruent': True, 'all_faces_nonplanar': True,
        'each_face_rigid_by_construction': True, 'internal_diagonal_is_not_a_hinge': True,
        'driver_interval_degrees': [0, MAX_DEGREES], 'sampled_configurations': samples,
        'side_length_exact': 'sqrt(126)/5', 'diagonal_lengths_exact': [4, 2],
        'six_squared_distances_in_order_AB_BC_CD_DA_AC_BD': ['126/25'] * 4 + ['16', '4'],
        'face_signed_six_volume_exact': '8/5',
        'max_pair_distance_error': max_distance_error,
        'max_anchor_coordinate_error': max_anchor_error,
        'max_signed_six_volume_error': max_volume_error,
        'all_16_vertex_signatures_preserved_at_samples': True,
        'vertex_signature_exact_lower_bound': '120/227',
        'vertex_signatures_preserved_on_full_interval_by_proof': True,
        'displayed_triangle_surface_embedded_on_full_interval_by_proof': True,
        'proof_file': 'proofs/5x5 nonplanar rigid screw folding.txt',
        'minimum_absolute_vertex_signature': minimum_signature,
        'all_40_hinges_move': True,
        'min_hinge_angle_change_degrees': float(hinge_change.min()),
        'max_hinge_angle_change_degrees': float(hinge_change.max()),
        'max_vertex_displacement': float(np.linalg.norm(final - initial, axis=-1).max()),
        'geometry_changed_from_previous_model': True,
        'not_claimed': 'No claim of flat folding, original initial pose, or unchanged original driver circle.'
    }
    return report


def add_command(construction, name, inputs, outputs):
    command = ET.SubElement(construction, 'command', name=name)
    ET.SubElement(command, 'input', {f'a{index}': str(value) for index, value in enumerate(inputs)})
    ET.SubElement(command, 'output', {f'a{index}': str(value) for index, value in enumerate(outputs)})


def add_numeric(construction, label, expression, value, slider=False):
    if expression is not None:
        ET.SubElement(construction, 'expression', label=label, exp=expression, type='numeric')
    element = ET.SubElement(construction, 'element', type='numeric', label=label)
    ET.SubElement(element, 'value', val=str(value))
    ET.SubElement(element, 'show', object=str(slider).lower(), label='true')
    ET.SubElement(element, 'objColor', r='184', g='62', b='69', alpha='0')
    ET.SubElement(element, 'layer', val='0')
    ET.SubElement(element, 'labelMode', val='0')
    ET.SubElement(element, 'auxiliary', val='false')
    if slider:
        ET.SubElement(element, 'slider', min='0', max=str(MAX_DEGREES), absoluteScreenLocation='true',
                      width='220', x='35', y='55', fixed='false', horizontal='true', showAlgebra='true')
        ET.SubElement(element, 'lineStyle', thickness='4', type='0', typeHidden='1')
        ET.SubElement(element, 'animation', step='0.1', speed='0.6', type='0', playing='false')
        ET.SubElement(element, 'caption', val='折叠参数（度）')


def build_ggb(output, thumbnail):
    source = PROJECT / 'geogebra/2x2 nonplanar folding.ggb'
    with ZipFile(source) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(contents['geogebra.xml'])
    construction = root.find('construction')
    templates = {element.get('label'): deepcopy(element) for element in construction.findall('element')}
    construction.clear()
    construction.set('title', '5x5 congruent nonplanar rigid folding - exact screw construction')
    vertices, values = geometry(0)
    add_numeric(construction, 'fold', None, 0., slider=True)
    add_numeric(construction, 'diagLong', '2', HALF_LONG)
    add_numeric(construction, 'diagShort', '1', HALF_SHORT)
    add_numeric(construction, 'offset', '1/5', OFFSET)
    formulas = {
        'theta': 'fold*pi/180',
        'transverseLong': 'diagLong*cos(theta)',
        'transverseShort': '-diagShort*sin(theta)',
        'axisShift': '(transverseLong^2-transverseShort^2-offset^2)/(2*offset)',
        'radius': 'sqrt(axisShift^2+transverseLong^2)',
        'halfSum': 'atan2(transverseLong,axisShift)',
        'halfDifference': 'atan2(transverseShort,axisShift+offset)',
        'rowTurn': 'halfSum+halfDifference',
        'columnTurn': 'halfSum-halfDifference',
        'rowRise': 'diagLong*sin(theta)+diagShort*cos(theta)',
        'columnRise': 'diagLong*sin(theta)-diagShort*cos(theta)',
    }
    for label, expression in formulas.items():
        add_numeric(construction, label, expression, values[label])
    grid_labels = [[f'P_{{{row}{column}}}' for column in range(6)] for row in range(6)]
    for row in range(6):
        for column in range(6):
            phase_label, height_label = f'phase_{{{row}{column}}}', f'height_{{{row}{column}}}'
            phase = (row - 2) * values['rowTurn'] + (column - 2) * values['columnTurn'] - values['halfSum']
            height = (row - 2) * values['rowRise'] + (column - 2) * values['columnRise']
            add_numeric(construction, phase_label, f'({row-2})*rowTurn+({column-2})*columnTurn-halfSum', phase)
            add_numeric(construction, height_label,
                        f'({row-2})*rowRise+({column-2})*columnRise-diagLong*sin(theta)', height)
            expression = (f'(-axisShift+radius*cos({phase_label}),'
                          f'radius*sin({phase_label})*cos(theta)+{height_label}*sin(theta),'
                          f'-radius*sin({phase_label})*sin(theta)+{height_label}*cos(theta))')
            label = grid_labels[row][column]
            ET.SubElement(construction, 'expression', label=label, exp=expression, type='point')
            point = deepcopy(templates['A'])
            point.set('label', label)
            point.find('coords').attrib.update(dict(zip(('x', 'y', 'z'), map(str, vertices[row, column]))), w='1')
            point.find('show').attrib.update(object='false', label='false')
            ET.SubElement(point, 'auxiliary', val='false')
            construction.append(point)
    for row in range(5):
        for column in range(5):
            corners = [grid_labels[row][column], grid_labels[row][column + 1],
                       grid_labels[row + 1][column + 1], grid_labels[row + 1][column]]
            for triangle_index, indices in enumerate(((0, 1, 2), (0, 2, 3))):
                label = f'panel_{{{row+1}{column+1}{triangle_index}}}'
                edge_labels = [f'inner_{{{row+1}{column+1}{triangle_index}{edge}}}' for edge in range(3)]
                add_command(construction, 'Polygon', [corners[index] for index in indices], [label, *edge_labels])
                face = deepcopy(templates['t5'])
                face.set('label', label)
                face.find('lineStyle').set('thickness', '0')
                face.find('show').attrib.update(object='true', label='false')
                color = [round(value * 255) for value in to_rgb(COLORS[row * 5 + column])]
                face.find('objColor').attrib.update(dict(zip(('r', 'g', 'b'), map(str, color))), alpha='1')
                ET.SubElement(face, 'auxiliary', val='false')
                construction.append(face)
                for edge_label in edge_labels:
                    edge = deepcopy(templates['f'])
                    edge.set('label', edge_label)
                    edge.find('show').attrib.update(object='false', label='false')
                    ET.SubElement(edge, 'auxiliary', val='false')
                    construction.append(edge)
    flat_labels = [label for row in grid_labels for label in row]
    for index, endpoints in enumerate(edge_incidence()):
        label = f'edge_{{{index+1}}}'
        add_command(construction, 'Segment', [flat_labels[endpoint] for endpoint in endpoints], [label])
        edge = deepcopy(templates['f'])
        edge.set('label', label)
        edge.find('show').attrib.update(object='true', label='false')
        edge.find('objColor').attrib.update(r='35', g='60', b='77')
        edge.find('lineStyle').set('thickness', '3')
        ET.SubElement(edge, 'auxiliary', val='false')
        construction.append(edge)
    root.find('kernel/angleUnit').set('val', 'radian')
    root.find('kernel/decimals').set('val', '6')
    root.find('kernel/algebraStyle').set('val', '3')
    root.find('algebraView/mode').set('val', '3')
    camera = root.find('euclidianView3D/coordSystem')
    camera.attrib.update(xZero='1', yZero='-1', zZero='0', scale='36', xAngle='12', zAngle='5')
    for view in root.findall('gui/perspectives/perspective/views/view'):
        view.set('visible', str(view.get('id') in ('2', '512')).lower())
    contents['geogebra.xml'] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    contents['geogebra_thumbnail.png'] = thumbnail
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    return grid_labels


def surfaces(axis, vertices):
    triangles = vertices.reshape(-1, 3)[TRIANGLE_IDS]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    light = np.array([1., -.35, .6])
    light /= np.linalg.norm(light)
    shade = (.75 + .25 * np.abs(normals @ light))[:, None]
    colors = np.repeat(np.array([to_rgb(color) for color in COLORS]), 2, axis=0) * shade
    mesh = Poly3DCollection(triangles, facecolors=colors, edgecolors=colors, linewidths=.25, antialiaseds=False)
    axis.add_collection3d(mesh)
    boundaries = vertices.reshape(-1, 3)[FACE_IDS[:, [0, 1, 2, 3, 0]]]
    lines = Line3DCollection(boundaries, colors='#233c4d', linewidths=1)
    axis.add_collection3d(lines)
    return mesh, lines


def setup_axis(axis, extent):
    lower, upper = extent
    center = (lower + upper) / 2
    spans = (upper - lower) * 1.12
    axis.set(xlim=(center[0] - spans[0] / 2, center[0] + spans[0] / 2),
             ylim=(center[1] - spans[1] / 2, center[1] + spans[1] / 2),
             zlim=(center[2] - spans[2] / 2, center[2] + spans[2] / 2))
    axis.set_box_aspect(spans, zoom=1.28)
    axis.view_init(elev=15, azim=7, roll=25)
    axis.set_proj_type('ortho')
    axis.set_axis_off()
    axis.set_facecolor('#f6f8fb')


def render(static_path, animation_path, frames=100):
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    degrees = MAX_DEGREES * (1 - np.cos(np.linspace(0, 2 * np.pi, frames))) / 2
    configurations = [geometry(value)[0] for value in degrees]
    points = np.array(configurations).reshape(-1, 3)
    extent = points.min(axis=0), points.max(axis=0)
    figure = plt.figure(figsize=(10, 8), facecolor='#f6f8fb')
    axis = figure.add_axes([.03, .12, .94, .77], projection='3d')
    initial, _ = geometry(0)
    surfaces(axis, initial)
    setup_axis(axis, extent)
    for index, panel in enumerate(initial.reshape(-1, 3)[FACE_IDS]):
        axis.text(*panel.mean(axis=0), f'{index//5+1},{index%5+1}', ha='center', va='center', fontsize=8,
                  color='#20394b', zorder=20, bbox={'fc': 'white', 'ec': 'none', 'alpha': .65, 'pad': 1})
    figure.suptitle('5 × 5 全等空间四边形 · 刚性拼接', fontsize=20, y=.97)
    figure.text(.5, .9, '新构型：25 个面全部非平面，40 条整边连接', ha='center', fontsize=11, color='#4c6376')
    figure.text(.5, .055, '四边 = √126 / 5；两条对角线 = 4、2；同一面的两片三角形无相对转动', ha='center', fontsize=10, color='#4c6376')
    figure.savefig(static_path, dpi=150)
    buffer = BytesIO()
    figure.savefig(buffer, format='png', dpi=35)
    plt.close(figure)
    thumbnail = Image.open(buffer)
    thumbnail.thumbnail((256, 256))
    packed = BytesIO()
    thumbnail.save(packed, format='PNG')
    figure = plt.figure(figsize=(14, 7), facecolor='#f6f8fb')
    fixed_axis = figure.add_axes([.01, .12, .49, .73], projection='3d')
    moving_axis = figure.add_axes([.5, .12, .49, .73], projection='3d')
    surfaces(fixed_axis, initial)
    setup_axis(fixed_axis, extent)
    setup_axis(moving_axis, extent)
    figure.suptitle('5 × 5 全等空间四边形 · 连续刚性折叠', fontsize=20, y=.97)
    figure.text(.25, .88, '初始拼接（固定视角）', ha='center', fontsize=12, color='#34556f')
    caption = figure.text(.75, .88, '', ha='center', fontsize=12, color='#34556f')
    figure.text(.5, .075, '中央面固定 · 25 面全等且不变形 · 40 条共边不分离', ha='center', fontsize=12, color='#34556f')
    figure.text(.5, .035, '解析刚体构造；并非把三角形间的对角线当作折痕', ha='center', fontsize=10, color='#526e83')

    def update(index):
        for collection in list(moving_axis.collections):
            collection.remove()
        surfaces(moving_axis, configurations[index])
        caption.set_text(f'折叠参数 {degrees[index]:.1f}°')
        return []

    animation = FuncAnimation(figure, update, frames=frames, interval=60, blit=False)
    animation.save(animation_path, writer=PillowWriter(fps=18), dpi=95)
    plt.close(figure)
    return packed.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--frames', type=int, default=100)
    arguments = parser.parse_args()
    report = validate()
    if not arguments.check_only:
        thumbnail = render(PROJECT / 'previews' / (STEM + '.png'), PROJECT / 'animations' / (STEM + '.gif'), arguments.frames)
        report['grid_labels'] = build_ggb(PROJECT / 'geogebra' / (STEM + '.ggb'), thumbnail)
        (PROJECT / 'geogebra' / (STEM + '.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
