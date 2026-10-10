"""Re-read the original GeoGebra circle construction and preserve it in a live check."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from audit_geogebra_skew_motion import eight_panel_patch, PANEL_DISTANCES
from build_geogebra_skew_grid5x5 import original_points, REFERENCE_ANGLE
from extend_geogebra_grid import coordinates


def circle_plane_intersections(center, radius, normal, plane_point, plane_normal):
    normals = np.array([normal, plane_normal], dtype=float)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    direction = np.cross(*normals)
    direction /= np.linalg.norm(direction)
    offsets = np.array([normals[0] @ center, normals[1] @ plane_point])
    base = normals.T @ np.linalg.solve(normals @ normals.T, offsets)
    middle = base + np.dot(center - base, direction) * direction
    height_squared = radius ** 2 - np.dot(middle - center, middle - center)
    if height_squared < -1e-12:
        raise ValueError('The source circle has no real intersection at this parameter.')
    offset = np.sqrt(max(height_squared, 0)) * direction
    return np.array([middle + offset, middle - offset])


def circle_circle_intersections(first_center, first_radius, first_normal,
                                second_center, second_radius, second_normal):
    candidates = circle_plane_intersections(first_center, first_radius, first_normal,
                                            second_center, second_normal)
    np.testing.assert_allclose(np.linalg.norm(candidates - second_center, axis=1),
                               second_radius, atol=1e-11, rtol=0)
    return candidates


def source_intersections(angle, previous):
    origin, first, second = np.zeros(3), np.array([2., 0., 0.]), np.array([0., 2., 0.])
    midpoint = (origin + second) / 2
    driven = midpoint + np.sqrt(3) * np.array([np.cos(angle), 0, np.sin(angle)])
    first_midpoint = first / 2
    follower_choices = circle_circle_intersections(
        origin, 2, driven, first_midpoint, np.sqrt(3), first - first_midpoint)
    follower = min(follower_choices, key=lambda candidate: np.linalg.norm(candidate - previous['G']))
    midpoint_diagonal = (driven + follower) / 2
    corner_choices = circle_circle_intersections(
        midpoint_diagonal, np.sqrt(2), driven - midpoint_diagonal,
        midpoint_diagonal, np.sqrt(2), midpoint_diagonal)
    corner = min(corner_choices, key=lambda candidate: np.linalg.norm(candidate - previous['K']))
    left_choices = circle_plane_intersections(driven, 2, driven, second, second)
    left = min(left_choices, key=lambda candidate: np.linalg.norm(candidate - previous['O']))
    right_choices = circle_circle_intersections(first, 2, first, follower, 2, follower)
    right = min(right_choices, key=lambda candidate: np.linalg.norm(candidate - previous['Q']))
    return {'A': origin, 'B': first, 'C': np.array([1., 1., -np.sqrt(2)]), 'D': second,
            'E': midpoint, 'F': driven, 'G': follower, 'K': corner, 'O': left, 'Q': right}


def validate_source(root):
    construction = root.find('construction')
    elements = {element.get('label'): element for element in construction.findall('element')}
    definitions = {command.find('output').get('a0'): command for command in construction.findall('command')}
    for label, command_name, inputs in (
            ('e', 'Circle', {'a0': 'E', 'a1': 'sqrt(3)', 'a2': 'd'}),
            ('F', 'Point', {'a0': 'e'}),
            ('G', 'Intersect', {'a0': 'p', 'a1': 'k'}),
            ('J', 'Intersect', {'a0': 'r', 'a1': 'q'}),
            ('O', 'Intersect', {'a0': 'r_1', 'a1': 't'}),
            ('Q', 'Intersect', {'a0': 'h_1', 'a1': 'e_1'})):
        assert definitions[label].get('name') == command_name
        assert definitions[label].find('input').attrib == inputs
    assert definitions['J'].find('output').get('a1') == 'K'
    assert elements['F'].find('fixed') is None or elements['F'].find('fixed').get('val') != 'true'
    saved = {label: coordinates(element) for label, element in elements.items()
             if element.get('type') in ('point', 'point3d')}
    np.testing.assert_allclose(saved['E'], (saved['A'] + saved['D']) / 2, atol=1e-12)
    angle = np.arctan2(saved['F'][2], saved['F'][0])
    independently_evaluated = source_intersections(angle, saved)
    for label, value in independently_evaluated.items():
        np.testing.assert_allclose(value, saved[label], atol=1e-12, rtol=0)
    maximum_distance_error, maximum_formula_error = 0.0, 0.0
    face_labels = (('O', 'D', 'A', 'F'), ('D', 'C', 'B', 'A'),
                   ('F', 'A', 'G', 'K'), ('A', 'B', 'Q', 'G'))
    for direction in (-1, 1):
        previous = saved
        for offset in np.linspace(0, direction * 10, 201):
            current_angle = angle + np.radians(offset)
            actual = source_intersections(current_angle, previous)
            expected = original_points(current_angle)
            maximum_formula_error = max(maximum_formula_error,
                max(float(np.linalg.norm(actual[label] - value)) for label, value in expected.items()))
            for face in face_labels:
                vertices = np.array([actual[label] for label in face])
                distances = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=-1)
                maximum_distance_error = max(maximum_distance_error,
                                             float(np.max(np.abs(distances - PANEL_DISTANCES))))
            previous = actual
    assert maximum_distance_error < 1e-11 and maximum_formula_error < 1e-11
    return {
        'source_F_definition': 'Point(e), not fixed',
        'source_circle_definition': 'Circle(E, sqrt(3), d)',
        'source_circle_center': saved['E'].tolist(),
        'source_circle_plane': 'y = 1',
        'source_circle_radius': 'sqrt(3)',
        'source_2x2_is_continuously_foldable_near_saved_pose': True,
        'independent_check': 'Circle/plane and circle/circle intersections, not extension formulas',
        'sample_count': 402,
        'maximum_original_four_panel_six_distance_error': maximum_distance_error,
        'maximum_difference_from_previous_analytical_source_model': maximum_formula_error,
    }


def point_expression(construction, template, label, expression, position):
    ET.SubElement(construction, 'expression', label=label, exp=expression, type='point')
    element = deepcopy(template)
    element.set('label', label)
    element.find('show').attrib.update(object='false', label='false')
    element.find('coords').attrib.update({axis: str(value) for axis, value in zip(('x', 'y', 'z'), position)})
    for child in list(element):
        if child.tag in ('animation', 'fixed', 'caption'):
            element.remove(child)
    ET.SubElement(element, 'auxiliary', val='true')
    construction.append(element)


def outside_expression(origin, first, second, incoming):
    axis = f'(Vector({origin},{first})+Vector({origin},{second}))'
    incoming_vector = f'Vector({origin},{incoming})'
    return f'{origin}-{incoming_vector}+2*({incoming_vector}*{axis})/({axis}*{axis})*{axis}'


def corner_expression(origin, first, second, long_diagonal, sign):
    first_vector, second_vector = f'Vector({origin},{first})', f'Vector({origin},{second})'
    factor = '(2/3)' if long_diagonal else '(1/2)'
    height = '(sqrt(2)/3)' if long_diagonal else '(1/(2*sqrt(2)))'
    return f'{origin}+{factor}*({first_vector}+{second_vector})+({sign})*{height}*Cross({first_vector},{second_vector})'


def build(source, output):
    if source.resolve() == output.resolve():
        raise ValueError('Never overwrite the supplied construction.')
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with ZipFile(source) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(contents['geogebra.xml'])
    report = validate_source(root)
    construction = root.find('construction')
    definitions = [ET.tostring(node) for node in construction if node.tag in ('expression', 'command')]
    elements = {element.get('label'): element for element in construction.findall('element')}
    vertices, signs = eight_panel_patch(REFERENCE_ANGLE)
    for element in elements.values():
        label = element.get('label')
        if element.get('type') in ('point', 'point3d'):
            element.find('show').attrib.update(object=str(label in ('A', 'F', 'K')).lower(),
                                               label=str(label in ('A', 'F', 'K')).lower())
        if label == 'F':
            element.find('coords').attrib.update(x='-1', y='1', z=str(np.sqrt(2)), w='1')
            element.find('pointSize').set('val', '7')
            element.find('objColor').attrib.update(r='190', g='44', b='66')
        if label in ('a_1', 'c_1', 'a_2', 'k_1', 'a_3', 'o', 'a_4', 'q_2'):
            element.find('show').set('object', 'false')
        if label in {f't{index}' for index in range(1, 9)}:
            element.find('objColor').set('alpha', '0.85')
            element.find('lineStyle').set('thickness', '0')
    point_expression(construction, elements['A'], 'T_{24}', outside_expression('B', 'C', 'Q', 'A'), vertices[1, 3])
    point_expression(construction, elements['A'], 'T_{42}', outside_expression('G', 'K', 'Q', 'A'), vertices[3, 1])
    recipes = (('T_{14}', 'B', 'C', 'T_{24}', False, (0, 3)),
               ('T_{34}', 'B', 'T_{24}', 'Q', True, (2, 3)),
               ('T_{41}', 'G', 'K', 'T_{42}', False, (3, 0)),
               ('T_{43}', 'G', 'T_{42}', 'Q', True, (3, 2)))
    for sign, (label, origin, first, second, long_diagonal, location) in zip(signs, recipes):
        point_expression(construction, elements['A'], label,
                         corner_expression(origin, first, second, long_diagonal, sign), vertices[location])
    new_panels = (('C', 'T_{14}', 'T_{24}', 'B'), ('B', 'T_{24}', 'T_{34}', 'Q'),
                  ('K', 'G', 'T_{42}', 'T_{41}'), ('G', 'Q', 'T_{43}', 'T_{42}'))
    for panel_index, panel in enumerate(new_panels):
        blue = panel_index in (0, 2)
        triangles = ((0, 1, 3), (1, 2, 3)) if blue else ((0, 1, 2), (0, 2, 3))
        for triangle_index, corners in enumerate(triangles):
            label = f'checkFace_{{{panel_index}{triangle_index}}}'
            edge_labels = [f'checkEdge_{{{panel_index}{triangle_index}{index}}}' for index in range(3)]
            command = ET.SubElement(construction, 'command', name='Polygon')
            ET.SubElement(command, 'input', {f'a{index}': panel[corner] for index, corner in enumerate(corners)})
            ET.SubElement(command, 'output', {f'a{index}': name for index, name in enumerate([label, *edge_labels])})
            face = deepcopy(elements['t5' if blue else 't1'])
            face.set('label', label)
            construction.append(face)
            diagonal = (1, 2)[triangle_index] if blue else (2, 0)[triangle_index]
            for index, edge_label in enumerate(edge_labels):
                edge = deepcopy(elements['d_1'])
                edge.set('label', edge_label)
                edge.find('show').attrib.update(object=str(index != diagonal).lower(), label='false')
                edge.find('objColor').attrib.update(r='46', g='69', b='89')
                construction.append(edge)
    command = ET.SubElement(construction, 'command', name='Segment')
    ET.SubElement(command, 'input', a0='T_{34}', a1='T_{43}')
    ET.SubElement(command, 'output', a0='checkDiagonal')
    segment = deepcopy(elements['d_1'])
    segment.set('label', 'checkDiagonal')
    segment.find('show').attrib.update(object='true', label='true')
    segment.find('objColor').attrib.update(r='190', g='44', b='66')
    segment.find('lineStyle').attrib.update(thickness='5', type='10')
    segment.find('labelMode').set('val', '1')
    construction.append(segment)
    camera = root.find('euclidianView3D/coordSystem')
    center = np.nanmean(vertices, axis=(0, 1))
    camera.attrib.update({axis + 'Zero': str(-value) for axis, value in zip(('x', 'y', 'z'), center)})
    camera.attrib.update(scale='65', xAngle='65', zAngle='108.434948822922')
    decimals = root.find('kernel/decimals')
    if decimals is not None:
        decimals.set('val', '6')
    construction.set('title', 'Original F driver preserved - eight-panel closure check, NOT a 5x5 result')
    retained = [ET.tostring(node) for node in construction if node.tag in ('expression', 'command')]
    assert retained[:len(definitions)] == definitions
    assert elements['F'].find('fixed') is None
    contents['geogebra.xml'] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    contents.pop('geogebra_thumbnail.png', None)
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    report.update(source_sha256=original_hash, original_definitions_preserved=len(definitions),
                  output_status='live_eight_panel_diagnostic_not_a_5x5_deliverable',
                  initial_F_in_diagnostic=[-1, 1, 'sqrt(2)'])
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(output.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('origami_skew_original_driver_check.ggb'))
    arguments = parser.parse_args()
    build(arguments.source, arguments.output)
