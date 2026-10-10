"""Add the requested rigid blue panel as an explicitly labelled joint-closure check."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from audit_geogebra_skew_motion import eight_panel_patch, opposite_corner, PANEL_DISTANCES
from build_geogebra_skew_grid5x5 import REFERENCE_ANGLE, reference_mesh
from verify_geogebra_skew_source import corner_expression, point_expression


def rigid_blue_panel(vertices):
    origin, hinge_end, target = vertices[2, 2], vertices[2, 3], vertices[3, 2]
    hinge = hinge_end - origin
    midpoint = (origin + hinge_end) / 2
    perpendicular = target - midpoint
    perpendicular -= np.dot(perpendicular, hinge) / np.dot(hinge, hinge) * hinge
    moving_corner = midpoint + np.sqrt(3) * perpendicular / np.linalg.norm(perpendicular)
    outer_corner = opposite_corner(origin, hinge_end, moving_corner, True, 1)
    return np.array([origin, hinge_end, outer_corner, moving_corner])


def validate():
    initial, signs = eight_panel_patch(REFERENCE_ANGLE)
    initial_panel = rigid_blue_panel(initial)
    _, reference, _, _ = reference_mesh()
    np.testing.assert_allclose(initial_panel, reference[[3, 3, 4, 4], [3, 4, 4, 3]], atol=1e-12, rtol=0)
    samples = []
    maximum_error = 0.0
    for offset in np.linspace(-10, 10, 401):
        vertices, _ = eight_panel_patch(REFERENCE_ANGLE + np.radians(offset), signs)
        panel = rigid_blue_panel(vertices)
        distances = np.linalg.norm(panel[:, None] - panel[None, :], axis=-1)
        error = float(np.max(np.abs(distances - PANEL_DISTANCES)))
        maximum_error = max(maximum_error, error)
        np.testing.assert_allclose(panel[:2], vertices[[2, 2], [2, 3]], atol=1e-12, rtol=0)
        if any(np.isclose(offset, value, atol=1e-10) for value in (-5, -1, 0, 1, 5)):
            samples.append({'circle_offset_degrees': float(offset),
                            'rigid_panel_distance_error': error,
                            'unattached_edge_endpoint_gap': float(np.linalg.norm(panel[3] - vertices[3, 2]))})
    assert maximum_error < 1e-11
    assert samples[2]['unattached_edge_endpoint_gap'] < 1e-12
    assert all(sample['unattached_edge_endpoint_gap'] > 1e-6
               for sample in samples if sample['circle_offset_degrees'])
    return initial_panel, {
        'status': 'rigid_ninth_panel_trial_one_hinge_attached_not_a_closed_mesh',
        'original_eight_panel_motion_unchanged': True,
        'blue_panel_preserves_all_six_distances': True,
        'attached_edge': ['Q', 'T_{34}'],
        'unattached_edge_target': ['Q', 'T_{43}'],
        'unattached_edge_actual': ['Q', 'blueJoin'],
        'placement': 'Closest rigid-panel corner to T_{43} while holding Q and T_{34} fixed',
        'maximum_blue_panel_distance_error': maximum_error,
        'samples': samples,
    }


def build(source, output):
    if source.resolve() == output.resolve():
        raise ValueError('Keep the original eight-panel file unchanged.')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    initial, report = validate()
    with ZipFile(source) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(contents['geogebra.xml'])
    construction = root.find('construction')
    definitions = [ET.tostring(node) for node in construction if node.tag in ('command', 'expression')]
    elements = {element.get('label'): element for element in construction.findall('element')}
    assert not any(label in elements for label in ('blueMid', 'blueJoin', 'blueOuter', 'joinGap'))
    point_expression(construction, elements['A'], 'blueMid', 'Midpoint(Q,T_{34})', (initial[0] + initial[1]) / 2)
    hinge = 'Vector(Q,T_{34})'
    target = 'Vector(blueMid,T_{43})'
    perpendicular = f'({target}-({target}*{hinge})/({hinge}*{hinge})*{hinge})'
    point_expression(construction, elements['A'], 'blueJoin',
                     f'blueMid+sqrt(3)*UnitVector({perpendicular})', initial[3])
    point_expression(construction, elements['A'], 'blueOuter',
                     corner_expression('Q', 'T_{34}', 'blueJoin', True, 1), initial[2])
    for triangle_index, labels in enumerate((('Q', 'T_{34}', 'blueJoin'),
                                             ('T_{34}', 'blueOuter', 'blueJoin'))):
        face_label = f'bluePanel_{triangle_index}'
        edges = [f'blueEdge_{{{triangle_index}{index}}}' for index in range(3)]
        command = ET.SubElement(construction, 'command', name='Polygon')
        ET.SubElement(command, 'input', {f'a{index}': label for index, label in enumerate(labels)})
        ET.SubElement(command, 'output', {f'a{index}': label for index, label in enumerate([face_label, *edges])})
        face = deepcopy(elements['t5'])
        face.set('label', face_label)
        construction.append(face)
        hidden_edge = (1, 2)[triangle_index]
        for index, label in enumerate(edges):
            edge = deepcopy(elements['d_1'])
            edge.set('label', label)
            edge.find('show').attrib.update(object=str(index != hidden_edge).lower(), label='false')
            edge.find('objColor').attrib.update(r='46', g='69', b='89')
            construction.append(edge)
    command = ET.SubElement(construction, 'command', name='Segment')
    ET.SubElement(command, 'input', a0='T_{43}', a1='blueJoin')
    ET.SubElement(command, 'output', a0='joinGap')
    gap = deepcopy(elements['d_1'])
    gap.set('label', 'joinGap')
    gap.find('show').attrib.update(object='true', label='false')
    gap.find('objColor').attrib.update(r='208', g='37', b='60')
    gap.find('lineStyle').attrib.update(thickness='7', type='0')
    construction.append(gap)
    for label, caption in (('Q', 'Q'), ('T_{34}', 'X'), ('T_{43}', 'Y')):
        element = elements[label]
        element.find('show').attrib.update(object='true', label='true')
        element.find('labelMode').set('val', '3')
        ET.SubElement(element, 'caption', val=caption)
    elements['T_{43}'].find('objColor').attrib.update(r='208', g='37', b='60')
    elements['checkDiagonal'].find('show').set('label', 'false')
    construction.set('title', 'Rigid blue ninth panel - one-hinge trial; not a closed folding grid')
    retained = [ET.tostring(node) for node in construction if node.tag in ('command', 'expression')]
    assert retained[:len(definitions)] == definitions
    contents['geogebra.xml'] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(output.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).parent
    parser.add_argument('--source', type=Path, default=directory / 'origami_skew_original_driver_check.ggb')
    parser.add_argument('--output', type=Path, default=directory / 'origami_skew_ninth_panel_check.ggb')
    arguments = parser.parse_args()
    build(arguments.source, arguments.output)
