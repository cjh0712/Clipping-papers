"""Build the original K-branch 5x5 reference assembly, without claiming a rigid flex."""

import argparse
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image

from extend_geogebra_grid import arc_points, coordinates


SIZE = 5
SOURCE_LABELS = {(1, 1): 'O', (1, 2): 'D', (1, 3): 'C',
                 (2, 1): 'F', (2, 2): 'A', (2, 3): 'B',
                 (3, 1): 'K', (3, 2): 'G', (3, 3): 'Q'}
REFERENCE_ANGLE = np.arccos(-1/np.sqrt(3))


def original_points(angle):
    points = arc_points(angle)
    first, second, driven, follower = [points[label] for label in ('B', 'D', 'F', 'G')]
    points.update(C=np.array([1., 1., -np.sqrt(2)]),
                  O=2/3*(second+driven)-np.sqrt(2)/3*np.cross(second, driven),
                  Q=2/3*(first+follower)+np.sqrt(2)/3*np.cross(first, follower),
                  K=(driven+follower)/2-np.cross(driven, follower)/(2*np.sqrt(2)))
    return points


def reference_mesh():
    base = np.array([[[-2, 2, 0], [0, 2, 0]], [[-1, 1, 1], [0, 0, 0]]])
    horizontal, vertical = np.array([3, -1, -1]), np.array([1, -3, 1])
    coefficients = np.array([[base[(row-1) % 2, (column-1) % 2]
                              + ((row-1)//2)*vertical + ((column-1)//2)*horizontal
                              for column in range(6)] for row in range(6)])
    vertices = coefficients*np.array([1., 1., np.sqrt(2)])
    labels = np.array([[SOURCE_LABELS.get((row, column), f'V_{{{row}{column}}}')
                        for column in range(6)] for row in range(6)])
    indices = np.arange(36).reshape(6, 6)
    faces = np.stack([indices[:-1, :-1], indices[:-1, 1:],
                      indices[1:, 1:], indices[1:, :-1]], axis=-1).reshape(25, 4)
    return coefficients, vertices, labels, faces


def audit(vertices, faces, saved):
    reference = original_points(REFERENCE_ANGLE)
    for location, label in SOURCE_LABELS.items():
        np.testing.assert_allclose(vertices[location], reference[label], atol=1e-12)
    saved_angle = np.arctan2(saved['F'][2], saved['F'][0])
    for label, point in original_points(saved_angle).items():
        np.testing.assert_allclose(point, saved[label], atol=1e-11)
    flat = vertices.reshape(-1, 3)
    panels = flat[faces]
    expected = np.array([[0, 2, np.sqrt(8), 2], [2, 0, 2, 2],
                         [np.sqrt(8), 2, 0, 2], [2, 2, 2, 0]])
    distances = np.linalg.norm(panels[:, :, None]-panels[:, None, :], axis=-1)
    np.testing.assert_allclose(distances, np.broadcast_to(expected, distances.shape), atol=1e-12)
    incidence, pairs = {}, set()
    for face in faces:
        for first, second in zip(face, np.roll(face, -1)):
            edge = tuple(sorted((int(first), int(second))))
            incidence[edge] = incidence.get(edge, 0)+1
        for first in range(4):
            for second in range(first+1, 4):
                pairs.add(tuple(sorted((int(face[first]), int(face[second])))))
    assert len(incidence) == 60 and list(incidence.values()).count(2) == 40
    assert len(np.unique(flat, axis=0)) == 36
    pairs = sorted(pairs)
    pinned = {8, 9, 14, 15}
    free = np.array([index for index in range(108) if index//3 not in pinned])
    jacobian = np.zeros((len(pairs), 108))
    for row, (first, second) in enumerate(pairs):
        jacobian[row, 3*first:3*first+3] = flat[first]-flat[second]
        jacobian[row, 3*second:3*second+3] = flat[second]-flat[first]
    reduced = jacobian[:, free]
    _, singular, right = np.linalg.svd(reduced, full_matrices=False)
    velocity = np.zeros(108)
    velocity[free] = right[-1]
    velocity = velocity.reshape(-1, 3)
    driven_index = 13
    expected_velocity = np.array([-np.sqrt(2), 0, -1])
    velocity *= np.dot(expected_velocity, velocity[driven_index])/np.dot(velocity[driven_index], velocity[driven_index])
    np.testing.assert_allclose(velocity[driven_index], expected_velocity, atol=1e-11)
    right_side = np.array([-np.dot(velocity[first]-velocity[second], velocity[first]-velocity[second])
                           for first, second in pairs])
    acceleration = np.linalg.lstsq(reduced, right_side, rcond=1e-10)[0]
    residual = reduced@acceleration-right_side
    return {
        'status': 'static_reference_only',
        'original_corner': 'K, not J',
        'mesh_faces': 25,
        'mesh_vertices': 36,
        'whole_edge_hinges': 40,
        'face_distance_error': float(np.max(np.abs(distances-expected))),
        'saved_source_circle_angle_degrees': float(np.degrees(saved_angle)),
        'reference_circle_angle_degrees': float(np.degrees(REFERENCE_ANGLE)),
        'reference_F': [-1, 1, 'sqrt(2)'],
        'free_coordinate_count': len(free),
        'distance_constraint_rank': int(np.sum(singular > 1e-9)),
        'first_order_distance_residual': float(np.linalg.norm(jacobian@velocity.ravel())),
        'second_order_distance_residual_l2': float(np.linalg.norm(residual)),
        'second_order_distance_residual_max': float(np.max(np.abs(residual))),
        'interpretation': 'This regular K-branch reference is congruent and edge-connected. Its nonzero '
                          'second-order residual rejects a smooth rigid motion with the original nonzero '
                          'circle velocity. This is not a proof about all other possible assemblies.',
    }


def preview(vertices, faces, output):
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    figure = plt.figure(figsize=(11, 9), facecolor='#f7f9fc')
    axis = figure.add_subplot(projection='3d', facecolor='#f7f9fc', computed_zorder=False)
    axis.set_position([0.02, 0.15, 0.96, 0.76])
    panels = vertices.reshape(-1, 3)[faces]
    triangles, colors, boundaries = [], [], []
    for index, panel in enumerate(panels):
        row, column = divmod(index, SIZE)
        blue = (row+column) % 2 == 0
        triangle_ids = ((0, 1, 3), (1, 2, 3)) if blue else ((0, 1, 2), (0, 2, 3))
        triangles.extend([panel[list(triangle)] for triangle in triangle_ids])
        colors.extend(['#75aeda' if blue else '#f0cd64']*2)
        boundaries.append(panel[[0, 1, 2, 3, 0]])
        center = panel.mean(axis=0)
        axis.text(*center, f'{row+1},{column+1}', ha='center', va='center', fontsize=9,
                  color='#102f49', zorder=5,
                  bbox={'boxstyle': 'round,pad=0.15', 'fc': 'white', 'ec': 'none', 'alpha': 0.75})
    triangle_array = np.array(triangles)
    normals = np.cross(triangle_array[:, 1]-triangle_array[:, 0], triangle_array[:, 2]-triangle_array[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    light = np.array([0.4, -0.2, 0.8])
    light /= np.linalg.norm(light)
    shaded = np.array([to_rgb(color) for color in colors])*(0.82+0.18*np.abs(normals@light))[:, None]
    axis.add_collection3d(Poly3DCollection(triangles, facecolors=shaded, edgecolors=shaded,
                                         antialiaseds=False, linewidths=0, alpha=1, zorder=1))
    axis.add_collection3d(Line3DCollection(boundaries, colors='#2e4559', linewidths=1.1, zorder=2))
    original_perimeter = vertices[[1, 1, 1, 2, 3, 3, 3, 2, 1], [1, 2, 3, 3, 3, 2, 1, 1, 1]]
    axis.plot(*original_perimeter.T, color='#b34036', linewidth=2.4, zorder=4)
    lower, upper = vertices.min(axis=(0, 1)), vertices.max(axis=(0, 1))
    padding = (upper-lower)*0.04
    axis.set(xlim=(lower[0]-padding[0], upper[0]+padding[0]),
             ylim=(lower[1]-padding[1], upper[1]+padding[1]),
             zlim=(lower[2]-padding[2], upper[2]+padding[2]))
    axis.set_box_aspect(upper-lower)
    axis.set_proj_type('ortho')
    axis.view_init(elev=70, azim=25, roll=130)
    axis.set_axis_off()
    figure.suptitle('5 × 5 空间四边形 · 原 K 拼接方向', fontsize=21, fontweight='bold', y=0.96)
    figure.text(0.5, 0.905, '五行五列，25 个全等单元；同一单元同色，隐藏内部三角形共边',
                ha='center', fontsize=11, color='#445a6f')
    figure.text(0.5, 0.10, '红框：原 2×2 的四个面，保留 K，不换成 J', ha='center', fontsize=12, color='#a9362d')
    figure.text(0.5, 0.065, '静态布局核对图 · 初态取原圆上的 F = (−1, 1, √2) · 不是已验证的可折叠模型',
                ha='center', fontsize=10, color='#52677a')
    figure.savefig(output, dpi=150, bbox_inches='tight')
    buffer = BytesIO()
    figure.savefig(buffer, format='png', dpi=40, bbox_inches='tight')
    plt.close(figure)
    thumbnail = Image.open(buffer)
    thumbnail.thumbnail((256, 256))
    packed = BytesIO()
    thumbnail.save(packed, format='PNG')
    return packed.getvalue()


def add_point(construction, template, label, weights, visible=False):
    horizontal, vertical, height = map(int, weights)
    expression = f'({horizontal},{vertical},{height}*sqrt(2))'
    ET.SubElement(construction, 'expression', label=label, exp=expression, type='point')
    element = deepcopy(template)
    element.set('label', label)
    element.find('coords').attrib.update(x=str(horizontal), y=str(vertical), z=str(height*np.sqrt(2)), w='1')
    element.find('show').attrib.update(object=str(visible).lower(), label=str(visible).lower())
    for child in list(element):
        if child.tag in ('animation', 'fixed', 'auxiliary', 'caption'):
            element.remove(child)
    ET.SubElement(element, 'fixed', val='true')
    if not visible:
        ET.SubElement(element, 'auxiliary', val='true')
    if label == 'F':
        element.find('objColor').attrib.update(r='193', g='64', b='50')
        ET.SubElement(element, 'caption', val='F（静态参照）')
        element.find('labelMode').set('val', '3')
    construction.append(element)


def build(source, output, check_only=False):
    if source.resolve() == output.resolve():
        raise ValueError('Write a separate reference file; never overwrite the source.')
    with ZipFile(source) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root = ET.fromstring(contents['geogebra.xml'])
    construction = root.find('construction')
    elements = {element.get('label'): element for element in construction.findall('element')}
    saved = {label: coordinates(element) for label, element in elements.items()
             if element.get('type') in ('point', 'point3d')}
    coefficients, vertices, labels, faces = reference_mesh()
    report = audit(vertices, faces, saved)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if check_only:
        return
    construction.clear()
    construction.set('title', '5x5 skew - original K orientation - STATIC REFERENCE ONLY')
    for row in range(6):
        for column in range(6):
            label = labels[row, column]
            add_point(construction, elements['A'], label, coefficients[row, column], label in ('F', 'K'))
    add_point(construction, elements['A'], 'E', [0, 1, 0])
    command = ET.SubElement(construction, 'command', name='Segment')
    ET.SubElement(command, 'input', a0='A', a1='D')
    ET.SubElement(command, 'output', a0='d_arc')
    segment = deepcopy(elements['d_1'])
    segment.set('label', 'd_arc')
    segment.find('show').attrib.update(object='false', label='false')
    construction.append(segment)
    command = ET.SubElement(construction, 'command', name='Circle')
    ET.SubElement(command, 'input', a0='E', a1='sqrt(3)', a2='d_arc')
    ET.SubElement(command, 'output', a0='e')
    circle = deepcopy(elements['e'])
    circle.find('show').attrib.update(object='true', label='false')
    construction.append(circle)
    original_perimeter = ('O', 'D', 'C', 'B', 'Q', 'G', 'K', 'F', 'O')
    original_edges = {frozenset(pair) for pair in zip(original_perimeter, original_perimeter[1:])}
    for row in range(SIZE):
        for column in range(SIZE):
            quad = [labels[location] for location in
                    ((row, column), (row, column+1), (row+1, column+1), (row+1, column))]
            blue = (row+column) % 2 == 0
            triangles = ((0, 1, 3), (1, 2, 3)) if blue else ((0, 1, 2), (0, 2, 3))
            for triangle_index, corner_ids in enumerate(triangles):
                label = f'Q_{{{row+1}{column+1}{triangle_index}}}'
                edge_labels = [f'b_{{{row}{column}{triangle_index}{index}}}' for index in range(3)]
                command = ET.SubElement(construction, 'command', name='Polygon')
                ET.SubElement(command, 'input', {f'a{index}': quad[corner] for index, corner in enumerate(corner_ids)})
                ET.SubElement(command, 'output', {f'a{index}': name for index, name in enumerate([label, *edge_labels])})
                face = deepcopy(elements['t5' if blue else 't1'])
                face.set('label', label)
                face.find('show').attrib.update(object='true', label='false')
                face.find('lineStyle').set('thickness', '0')
                face.find('objColor').set('alpha', '0.85')
                construction.append(face)
                diagonal_index = (1, 2)[triangle_index] if blue else (2, 0)[triangle_index]
                for edge_index, edge_label in enumerate(edge_labels):
                    edge = deepcopy(elements['d_1'])
                    edge.set('label', edge_label)
                    edge.find('show').attrib.update(object=str(edge_index != diagonal_index).lower(), label='false')
                    edge.find('objColor').attrib.update(r='46', g='69', b='89')
                    edge.find('lineStyle').set('thickness', '3')
                    endpoints = frozenset((quad[corner_ids[edge_index]], quad[corner_ids[(edge_index+1) % 3]]))
                    if endpoints in original_edges:
                        edge.find('objColor').attrib.update(r='179', g='64', b='54')
                        edge.find('lineStyle').set('thickness', '5')
                    construction.append(edge)
    center = vertices.mean(axis=(0, 1))
    camera = root.find('euclidianView3D/coordSystem')
    camera.attrib.update({axis+'Zero': str(-value) for axis, value in zip(('x', 'y', 'z'), center)})
    camera.attrib.update(scale='40', xAngle=str(np.degrees(np.arctan2(6, np.sqrt(10)))),
                         zAngle=str(np.degrees(np.arctan2(3, -1))))
    assert len([element for element in construction.findall('element') if element.get('type') == 'polygon3d']) == 50
    assert 'J' not in set(labels.ravel())
    contents['geogebra.xml'] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    contents['geogebra_thumbnail.png'] = preview(vertices, faces, output.with_suffix('.png'))
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Static reference: {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'geogebra' / '5x5 nonplanar static.ggb')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    build(args.source, args.output, args.check_only)


if __name__ == '__main__':
    main()
