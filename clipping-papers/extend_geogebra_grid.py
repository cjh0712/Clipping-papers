"""Extend the supplied exact GeoGebra 2x2 construction to a shared-vertex 5x5."""

import argparse
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image


SIZE = 5
BASIS_LABELS = ('B', 'D', 'F', 'G')
ORIGINAL_POINTS = {(1, 1): "A'", (1, 2): 'D', (1, 3): 'C',
                   (2, 1): 'F', (2, 2): 'A', (2, 3): 'B',
                   (3, 1): "A'_{2}", (3, 2): 'G', (3, 3): "A'_{1}"}
ORIGINAL_FACES = {(1, 1): 't1', (1, 2): 'q1', (2, 1): 'q3', (2, 2): 'q2'}
VERTEX_IDS = np.arange(36).reshape(6, 6)
FACE_IDS = np.stack([VERTEX_IDS[:-1, :-1], VERTEX_IDS[1:, :-1],
                     VERTEX_IDS[1:, 1:], VERTEX_IDS[:-1, 1:]], axis=-1).reshape(25, 4)
RHOMBI = (np.indices((SIZE, SIZE)).sum(axis=0) % 2 == 0).ravel()


def coefficients():
    """Integer translations: column steps B,-F and row steps G,-D, anchored at A."""
    row_steps = np.array([[0, 0, 0, 1] if index % 2 == 0 else [0, -1, 0, 0]
                          for index in range(SIZE)])
    column_steps = np.array([[1, 0, 0, 0] if index % 2 == 0 else [0, 0, -1, 0]
                             for index in range(SIZE)])
    rows = np.vstack([np.zeros(4, dtype=int), np.cumsum(row_steps, axis=0)])
    columns = np.vstack([np.zeros(4, dtype=int), np.cumsum(column_steps, axis=0)])
    return rows[:, None] - rows[2] + columns[None, :] - columns[2]


COEFFICIENTS = coefficients()


def coordinates(element):
    values = element.find('coords').attrib
    if element.get('type') == 'point':
        return np.array([float(values['x']), float(values['y']), 0]) / float(values['z'])
    return np.array([float(values[axis]) for axis in ('x', 'y', 'z')]) / float(values['w'])


def vertices_from_points(points):
    basis = np.array([points[label] - points['A'] for label in BASIS_LABELS])
    return points['A'] + COEFFICIENTS @ basis


def expression_for(weights):
    expression = 'A'
    for weight, label in zip(weights, BASIS_LABELS):
        if weight:
            expression += f' {"+" if weight > 0 else "-"} {abs(int(weight))} * ({label} - A)'
    return expression


def arc_points(angle):
    """Numerically audit the source's two-circle intersection; not used in the .ggb."""
    driven = np.array([np.sqrt(3) * np.cos(angle), 1, np.sqrt(3) * np.sin(angle)])
    axis = driven[1:]
    square_norm = np.dot(axis, axis)
    middle = -driven[0] * axis / square_norm
    offset = np.sqrt(max(0, 3 - driven[0] ** 2 / square_norm))
    second = middle + offset * np.array([-axis[1], axis[0]]) / np.sqrt(square_norm)
    return {'A': np.zeros(3), 'B': np.array([2., 0, 0]), 'D': np.array([0, 2., 0]),
            'F': driven, 'G': np.array([1, *second])}


def validate(root, points):
    """Check source registration, all face distances, planarity and shared topology."""
    initial = vertices_from_points(points)
    for location, label in ORIGINAL_POINTS.items():
        np.testing.assert_allclose(initial[location], points[label], atol=1e-10, rtol=0)
    incidence = {}
    for indices in FACE_IDS:
        for first, second in zip(indices, np.roll(indices, -1)):
            edge = tuple(sorted((int(first), int(second))))
            incidence[edge] = incidence.get(edge, 0) + 1
    assert len(incidence) == 60 and list(incidence.values()).count(2) == 40
    faces = initial.reshape(-1, 3)[FACE_IDS]
    reference = np.linalg.norm(faces[:, :, None] - faces[:, None, :], axis=-1)
    worst = 0.0
    for angle in np.radians(np.linspace(1, 359, 721)):
        controls = arc_points(angle)
        vertices = vertices_from_points(controls)
        current = vertices.reshape(-1, 3)[FACE_IDS]
        distances = np.linalg.norm(current[:, :, None] - current[:, None, :], axis=-1)
        worst = max(worst, float(np.max(np.abs(distances - reference))))
        row_edges = current[:, 1] - current[:, 0]
        column_edges = current[:, 3] - current[:, 0]
        np.testing.assert_allclose(np.linalg.norm(row_edges, axis=1), 2, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(column_edges, axis=1), 2, atol=1e-10)
        np.testing.assert_allclose(np.sum(row_edges * column_edges, axis=1),
                                   np.where(RHOMBI, 2, 0), atol=1e-10)
        np.testing.assert_allclose(current[:, 2], current[:, 1] + current[:, 3] - current[:, 0], atol=1e-10)
    if worst > 1e-9:
        raise AssertionError(f'Rigid-panel error: {worst}')
    commands = root.find('construction').findall('command')
    source_polygons = {command.find('output').get('a0'): list(command.find('input').attrib.values())
                       for command in commands if command.get('name') == 'Polygon'}
    for location, label in ORIGINAL_FACES.items():
        face = faces[location[0] * SIZE + location[1]]
        source = np.array([points[point] for point in source_polygons[label]])
        np.testing.assert_allclose(np.min(np.linalg.norm(face[:, None] - source[None, :], axis=2), axis=1),
                                   0, atol=1e-10)
    print(f'PASS: 721 circle samples; 25 rigid planar faces; 36 shared vertices; 40 full-edge hinges; '
          f'max pair-distance error {worst:.3e}.')
    print('Original four faces and original F/e/G constructions are retained exactly.')
    return initial


def preview(vertices, output):
    figure = plt.figure(figsize=(9, 8), facecolor='white')
    axis = figure.add_subplot(projection='3d')
    faces = vertices.reshape(-1, 3)[FACE_IDS]
    axis.add_collection3d(Poly3DCollection(faces, facecolors=np.where(RHOMBI, '#1565c0', '#ffcc02'),
                                         edgecolors='#25374b', linewidths=1.0, alpha=0.85))
    circle_angle = np.linspace(0, 2 * np.pi, 181)
    circle = np.column_stack([np.sqrt(3) * np.cos(circle_angle), np.ones_like(circle_angle),
                              np.sqrt(3) * np.sin(circle_angle)])
    axis.plot(*circle.T, color='#cc3344', linestyle='--', linewidth=1.2)
    driven = vertices[2, 1]
    axis.scatter(*driven, color='#cc3344', s=35)
    axis.text(*driven, ' F', color='#cc3344', fontsize=12)
    lower, upper = vertices.min(axis=(0, 1)), vertices.max(axis=(0, 1))
    axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
    axis.set_box_aspect(upper - lower)
    axis.set_proj_type('ortho')
    axis.view_init(elev=35, azim=45)
    axis.set_axis_off()
    figure.suptitle('5 × 5 · Original GeoGebra Circle Driver', fontsize=17, fontweight='bold')
    figure.text(0.5, 0.04, '13 rhombi + 12 squares · 40 full-edge hinges · Drag F on the original circle',
                ha='center', fontsize=10, color='#46566a')
    figure.savefig(output, dpi=140, bbox_inches='tight')
    buffer = BytesIO()
    figure.savefig(buffer, format='png', dpi=45, bbox_inches='tight')
    plt.close(figure)
    thumbnail = Image.open(buffer)
    thumbnail.thumbnail((256, 256))
    packed = BytesIO()
    thumbnail.save(packed, format='PNG')
    return packed.getvalue()


def extend(source, output):
    if source.resolve() == output.resolve():
        raise ValueError('The output must be a new file, separate from the supplied 2x2.')
    with ZipFile(source) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root = ET.fromstring(contents['geogebra.xml'])
    construction = root.find('construction')
    elements = {element.get('label'): element for element in construction.findall('element')}
    points = {label: coordinates(element) for label, element in elements.items()
              if element.get('type') in ('point', 'point3d')}
    vertices = validate(root, points)
    original_definitions = [ET.tostring(node) for node in construction if node.tag in ('expression', 'command')]
    construction.set('title', '5x5 planar - original circle driver')
    labels = {}
    for row in range(6):
        for column in range(6):
            location = (row, column)
            label = ORIGINAL_POINTS.get(location, f'P_{{{row}{column}}}')
            labels[location] = label
            if location in ORIGINAL_POINTS:
                continue
            ET.SubElement(construction, 'expression', label=label,
                          exp=expression_for(COEFFICIENTS[location]), type='point')
            element = deepcopy(elements['A'])
            element.set('label', label)
            element.find('show').set('object', 'false')
            element.find('show').set('label', 'false')
            element.find('coords').attrib.update(dict(zip(('x', 'y', 'z'), map(str, vertices[location]))))
            ET.SubElement(element, 'auxiliary', val='true')
            construction.append(element)
    for row in range(SIZE):
        for column in range(SIZE):
            if (row, column) in ORIGINAL_FACES:
                continue
            point_labels = [labels[location] for location in
                            ((row, column), (row + 1, column), (row + 1, column + 1), (row, column + 1))]
            polygon_label = f'Q_{{{row + 1}{column + 1}}}'
            edge_labels = [f's_{{{row + 1}{column + 1}{index}}}' for index in range(4)]
            command = ET.SubElement(construction, 'command', name='Polygon')
            ET.SubElement(command, 'input', {f'a{index}': label for index, label in enumerate(point_labels)})
            ET.SubElement(command, 'output', {f'a{index}': label for index, label
                                             in enumerate([polygon_label, *edge_labels])})
            element = deepcopy(elements['t1' if (row + column) % 2 == 0 else 'q1'])
            element.set('label', polygon_label)
            construction.append(element)
            for label in edge_labels:
                edge = deepcopy(elements['b'])
                edge.set('label', label)
                construction.append(edge)
    for element in construction.findall('element'):
        label = element.get('label')
        if element.get('type') in ('point', 'point3d'):
            element.find('show').set('object', 'true' if label in ('A', 'F') else 'false')
            element.find('show').set('label', 'true' if label in ('A', 'F') else 'false')
        if label in ('q4', 's', 'g_3', 'k_1', "g'"):
            element.find('show').set('object', 'false')
        if label == 'F':
            element.find('pointSize').set('val', '7')
            element.find('objColor').attrib.update(r='204', g='51', b='68')
    center = vertices.mean(axis=(0, 1))
    camera = root.find('euclidianView3D/coordSystem')
    camera.attrib.update({axis + 'Zero': str(-value) for axis, value in zip(('x', 'y', 'z'), center)})
    camera.set('scale', '38')
    visible_faces = [element for element in construction.findall('element')
                     if element.get('type') == 'polygon3d' and element.find('show').get('object') == 'true']
    assert len(visible_faces) == 25
    retained_definitions = [ET.tostring(node) for node in list(construction)
                            if node.tag in ('expression', 'command')][:len(original_definitions)]
    assert retained_definitions == original_definitions
    contents['geogebra.xml'] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    contents['geogebra_thumbnail.png'] = preview(vertices, output.with_suffix('.png'))
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    print(f'GeoGebra file: {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'geogebra' / '5x5 planar folding.ggb')
    args = parser.parse_args()
    extend(args.source, args.output)


if __name__ == '__main__':
    main()
