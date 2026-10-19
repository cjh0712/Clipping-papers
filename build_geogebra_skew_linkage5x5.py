"""Extend the source with midpoint/reflection constructions, without rigid-panel constraints."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from build_geogebra_skew_grid5x5 import original_points, REFERENCE_ANGLE, SOURCE_LABELS
from extend_geogebra_grid import coordinates


def add_command(construction, name, inputs, outputs):
    command = ET.SubElement(construction, 'command', name=name)
    ET.SubElement(command, 'input', {f'a{index}': label for index, label in enumerate(inputs)})
    ET.SubElement(command, 'output', {f'a{index}': label for index, label in enumerate(outputs)})


def build(source, reference_file, output):
    if output.resolve() in (source.resolve(), reference_file.resolve()):
        raise ValueError('Write a new linkage file, not either input file.')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with ZipFile(source) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(contents['geogebra.xml'])
    construction = root.find('construction')
    definitions = [ET.tostring(node) for node in construction if node.tag in ('command', 'expression')]
    elements = {element.get('label'): element for element in construction.findall('element')}
    point_template = deepcopy(elements['A'])
    edge_template = deepcopy(elements['d_1'])
    face_templates = {True: deepcopy(elements['t5']), False: deepcopy(elements['t1'])}
    positions = original_points(REFERENCE_ANGLE)
    grid = [[SOURCE_LABELS.get((row, column), f'W_{{{row}{column}}}')
             for column in range(6)] for row in range(6)]
    generated_points = []

    def add_point(label, command, inputs, position):
        assert label not in elements and label not in positions
        add_command(construction, command, inputs, [label])
        point = deepcopy(point_template)
        point.set('label', label)
        point.find('show').attrib.update(object='false', label='false')
        point.find('coords').attrib.update({axis: str(value) for axis, value in zip(('x', 'y', 'z'), position)})
        for child in list(point):
            if child.tag in ('fixed', 'animation', 'auxiliary', 'caption'):
                point.remove(child)
        ET.SubElement(point, 'auxiliary', val='false')
        construction.append(point)
        positions[label] = position
        generated_points.append(label)

    def extend_line(labels, prefix):
        first, middle, last = labels[1:4]
        first_midpoint, last_midpoint = f'M_{{{prefix}12}}', f'M_{{{prefix}23}}'
        add_point(first_midpoint, 'Midpoint', [first, middle], (positions[first] + positions[middle]) / 2)
        add_point(last_midpoint, 'Midpoint', [middle, last], (positions[middle] + positions[last]) / 2)
        for index, original, center in ((0, last, first_midpoint), (4, first, last_midpoint), (5, first, last)):
            add_point(labels[index], 'Mirror', [original, center], 2 * positions[center] - positions[original])

    for element in elements.values():
        show = element.find('show')
        if show is not None:
            show.attrib.update(object='false', label='false')
    driven = elements['F']
    driven.find('coords').attrib.update(x='-1', y='1', z=str(np.sqrt(2)), w='1')
    driven.find('show').attrib.update(object='true', label='true')
    driven.find('pointSize').set('val', '8')
    driven.find('objColor').attrib.update(r='204', g='48', b='66')
    driven.find('animation').set('playing', 'false')
    elements['e'].find('show').set('object', 'true')
    elements['e'].find('objColor').attrib.update(r='191', g='70', b='86')
    elements['e'].find('lineStyle').attrib.update(thickness='2', type='10')

    for row in range(1, 4):
        extend_line(grid[row], f'H{row}')
    for column in range(6):
        extend_line([grid[row][column] for row in range(6)], f'V{column}')

    face_labels, boundary_incidence = [], Counter()
    for row in range(5):
        for column in range(5):
            quad = [grid[row][column], grid[row][column + 1],
                    grid[row + 1][column + 1], grid[row + 1][column]]
            boundary_incidence.update(tuple(sorted(pair)) for pair in zip(quad, quad[1:] + quad[:1]))
            blue = (row + column) % 2 == 0
            triangles = ((0, 1, 3), (1, 2, 3)) if blue else ((0, 1, 2), (0, 2, 3))
            for triangle_index, corners in enumerate(triangles):
                label = f'linkedFace_{{{row}{column}{triangle_index}}}'
                outputs = [label, *[f'linkedHidden_{{{row}{column}{triangle_index}{index}}}' for index in range(3)]]
                add_command(construction, 'Polygon', [quad[index] for index in corners], outputs)
                face = deepcopy(face_templates[blue])
                face.set('label', label)
                face.find('show').attrib.update(object='true', label='false')
                face.find('lineStyle').set('thickness', '0')
                color = ('72', '151', '211') if blue else ('236', '196', '73')
                face.find('objColor').attrib.update(dict(zip(('r', 'g', 'b'), color)), alpha='0.96')
                construction.append(face)
                face_labels.append(label)
                for edge_label in outputs[1:]:
                    edge = deepcopy(edge_template)
                    edge.set('label', edge_label)
                    edge.find('show').attrib.update(object='false', label='false')
                    construction.append(edge)
    for index, endpoints in enumerate(boundary_incidence):
        label = f'linkedBoundary_{{{index}}}'
        add_command(construction, 'Segment', endpoints, [label])
        edge = deepcopy(edge_template)
        edge.set('label', label)
        edge.find('show').attrib.update(object='true', label='false')
        edge.find('objColor').attrib.update(r='46', g='69', b='89')
        edge.find('lineStyle').set('thickness', '3')
        construction.append(edge)

    vertices = np.array([[positions[label] for label in row] for row in grid])
    with ZipFile(reference_file) as archive:
        reference_root = ET.fromstring(archive.read('geogebra.xml'))
    reference_points = {element.get('label'): coordinates(element)
                        for element in reference_root.findall('construction/element')
                        if element.get('type') in ('point', 'point3d')}
    reference = np.array([[reference_points[SOURCE_LABELS.get((row, column), f'V_{{{row}{column}}}')]
                           for column in range(6)] for row in range(6)])
    np.testing.assert_allclose(vertices, reference, atol=1e-12, rtol=0)
    assert len(set(label for row in grid for label in row)) == 36
    assert len(face_labels) == 50 and len(boundary_incidence) == 60
    assert sum(count == 2 for count in boundary_incidence.values()) == 40
    assert driven.find('fixed') is None
    retained = [ET.tostring(node) for node in construction if node.tag in ('command', 'expression')]
    assert retained[:len(definitions)] == definitions
    camera = root.find('euclidianView3D/coordSystem')
    camera.attrib.update({axis + 'Zero': str(-value) for axis, value in zip(('x', 'y', 'z'), vertices.mean(axis=(0, 1)))})
    camera.attrib.update(scale='43', xAngle='45', zAngle='45')
    for view in root.findall('gui/perspectives/perspective/views/view'):
        view.set('visible', str(view.get('id') in ('2', '512')).lower())
    root.find('kernel/algebraStyle').set('val', '3')
    root.find('algebraView/mode').set('val', '3')
    construction.set('title', '5x5 shared-vertex geometric linkage - no rigid-panel constraints')
    contents['geogebra.xml'] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    contents.pop('geogebra_thumbnail.png', None)
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    report = {
        'status': 'geometric_linkage_not_a_verified_rigid_fold',
        'source_definitions_preserved': len(definitions),
        'source_sha256': source_hash,
        'initial_grid_matches_approved_static_file': True,
        'grid_labels': grid,
        'triangle_labels': face_labels,
        'quad_count': 25,
        'grid_vertex_count': 36,
        'unique_boundary_segments': 60,
        'full_shared_edges': 40,
        'new_point_commands': ['Midpoint', 'Mirror'],
        'new_midpoints': 18,
        'new_grid_points': 27,
        'F_definition': 'Original Point(e); original Circle(E, sqrt(3), d)',
        'rigid_constraints_applied_to_extension': False,
        'initial_grid_coordinates': vertices.tolist(),
    }
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    output.with_suffix('.md').write_text(
        '# 5×5 空间四边形几何联动\n\n'
        '- 打开同名 `.ggb`，拖动红色 F 点；网页预览也提供播放和复位。\n'
        '- 保留原始 2×2、K 和 F 的原圆，初态与已确认的 5×5 静态图一致。\n'
        '- 其余点用取中点、关于点对称补齐；25 个面使用 36 个共享顶点、40 条共边。\n'
        '- 隐藏面内三角形的共边，同一空间四边形使用同色。\n'
        '- 新增面片未施加刚性约束，运动中可能变形；这不是精确刚性折叠的证明。\n', encoding='utf-8')
    print(output.resolve())
    print('25 faces; 36 shared grid vertices; 40 shared edges; original 48 definitions retained.')
    print('Initial arrangement matches the approved static file. No rigidity acceptance test applied.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).parent
    parser.add_argument('source', type=Path)
    parser.add_argument('--reference', type=Path, default=directory / 'geogebra' / '5x5 nonplanar static.ggb')
    parser.add_argument('--output', type=Path, default=directory / 'geogebra' / '5x5 nonplanar folding.ggb')
    arguments = parser.parse_args()
    build(arguments.source, arguments.reference, arguments.output)
