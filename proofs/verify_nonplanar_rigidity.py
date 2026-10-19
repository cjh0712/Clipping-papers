"""Verify the exact rigidity certificate for this specific nonplanar 5x5 assembly."""

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import sympy as sp


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_LABELS = {(1, 1): 'O', (1, 2): 'D', (1, 3): 'C',
                 (2, 1): 'F', (2, 2): 'A', (2, 3): 'B',
                 (3, 1): 'K', (3, 2): 'G', (3, 3): 'Q'}
VELOCITIES = [(0, 0, -2), (0, 0, 0), (0, 0, 0), (0, -2, -1),
              (-2, 0, -1), (0, 0, 0), (0, 0, 0), (-2, -2, 0),
              (0, 0, 0), (0, 2, 1), (0, 0, 2), (0, 0, 0),
              (0, 4, 2), (2, 2, 4), (2, 4, 3), (0, 0, 4)]
STRESS = [(2, 5, 1), (3, 8, 1), (5, 6, -1), (5, 9, 1),
          (6, 10, 1), (6, 11, -2), (7, 8, -1), (7, 10, -2),
          (7, 11, 1), (8, 12, 1), (9, 10, -1), (9, 14, 1),
          (10, 11, 1), (10, 14, -1), (11, 12, -1), (11, 15, -1),
          (12, 15, 1), (14, 15, 1)]
EXTENSION = [
    ((0, 2), [(1, 1), (1, 2), (1, 3)]),
    ((0, 3), [(1, 2), (1, 3), (1, 4)]),
    ((5, 2), [(4, 1), (4, 2), (4, 3)]),
    ((5, 3), [(4, 2), (4, 3), (4, 4)]),
    ((2, 0), [(1, 1), (2, 1), (3, 1)]),
    ((3, 0), [(2, 1), (3, 1), (4, 1)]),
    ((2, 5), [(1, 4), (2, 4), (3, 4)]),
    ((3, 5), [(2, 4), (3, 4), (4, 4)]),
    ((0, 1), [(1, 1), (1, 2), (0, 2)]),
    ((0, 4), [(1, 3), (1, 4), (0, 3)]),
    ((5, 1), [(4, 1), (4, 2), (5, 2)]),
    ((5, 4), [(4, 3), (4, 4), (5, 3)]),
    ((1, 0), [(1, 1), (2, 1), (2, 0)]),
    ((4, 0), [(3, 1), (4, 1), (3, 0)]),
    ((1, 5), [(1, 4), (2, 4), (2, 5)]),
    ((4, 5), [(3, 4), (4, 4), (3, 5)]),
    ((0, 0), [(0, 1), (1, 0), (1, 1)]),
    ((0, 5), [(0, 4), (1, 4), (1, 5)]),
    ((5, 0), [(4, 0), (4, 1), (5, 1)]),
    ((5, 5), [(4, 4), (4, 5), (5, 4)]),
]


def read_xml(path):
    with ZipFile(path) as archive:
        return ET.fromstring(archive.read('geogebra.xml'))


def definitions(root):
    return [ET.tostring(node) for node in root.find('construction')
            if node.tag in ('command', 'expression')]


def exact_geometry(path):
    root = read_xml(path)
    expressions = {node.get('label'): node.get('exp') for node in root.findall('construction/expression')}
    points = {}
    base = [[(-2, 2, 0), (0, 2, 0)], [(-1, 1, 1), (0, 0, 0)]]
    for row in range(6):
        for column in range(6):
            label = SOURCE_LABELS.get((row, column), f'V_{{{row}{column}}}')
            match = re.fullmatch(r'\((-?\d+),(-?\d+),(-?\d+)\*sqrt\(2\)\)', expressions[label])
            assert match, (label, expressions[label])
            points[row, column] = sp.Matrix(list(map(int, match.groups())))
            expected = (sp.Matrix(base[(row - 1) % 2][(column - 1) % 2])
                        + ((row - 1) // 2) * sp.Matrix([1, -3, 1])
                        + ((column - 1) // 2) * sp.Matrix([3, -1, -1]))
            assert points[row, column] == expected
    triangles = {command.find('output').get('a0'): set(command.find('input').attrib.values())
                 for command in root.findall('construction/command') if command.get('name') == 'Polygon'}
    metric = sp.diag(1, 1, 2)
    full_pairs = set()
    for row in range(5):
        for column in range(5):
            locations = [(row, column), (row, column + 1), (row + 1, column + 1), (row + 1, column)]
            names = {SOURCE_LABELS.get(location, f'V_{{{location[0]}{location[1]}}}') for location in locations}
            combined = triangles[f'Q_{{{row + 1}{column + 1}0}}'] | triangles[f'Q_{{{row + 1}{column + 1}1}}']
            assert combined == names
            differences = [points[location] - points[locations[0]] for location in locations[1:]]
            assert sp.Matrix.hstack(*differences).det() != 0
            distances = []
            for first, second in combinations(locations, 2):
                difference = points[first] - points[second]
                distances.append((difference.T * metric * difference)[0])
                full_pairs.add(tuple(sorted((first, second))))
            assert sorted(distances) == [4, 4, 4, 4, 4, 8]
    assert len(full_pairs) == 110
    return points, full_pairs


def verify(static_path):
    points, full_pairs = exact_geometry(static_path)
    patch = [points[row, column] for row in range(1, 5) for column in range(1, 5)]
    pairs = sorted({tuple(sorted(pair)) for row in range(3) for column in range(3)
                    for pair in combinations((4 * row + column, 4 * row + column + 1,
                                               4 * (row + 1) + column, 4 * (row + 1) + column + 1), 2)})
    assert len(pairs) == 42
    free = [index for index in range(48) if index // 3 not in (1, 2, 5, 6)]
    metric = sp.diag(1, 1, 2)
    full_matrix = sp.zeros(42, 48)
    for row, (first, second) in enumerate(pairs):
        difference = metric * (patch[first] - patch[second])
        for axis in range(3):
            full_matrix[row, 3 * first + axis] = difference[axis]
            full_matrix[row, 3 * second + axis] = -difference[axis]
    matrix = full_matrix[:, free]
    velocity = sp.Matrix([value for point in VELOCITIES for value in point])
    reduced_velocity = velocity[free, :]
    assert matrix * reduced_velocity == sp.zeros(42, 1)
    assert reduced_velocity != sp.zeros(36, 1)
    excluded_rows = {4, 6, 7, 9, 10, 17, 41}
    minor_rows = [index for index in range(42) if index + 1 not in excluded_rows]
    determinant = matrix.extract(minor_rows, list(range(35))).det(method='domain-ge')
    assert determinant == -(2 ** 36)
    stress_map = {(first - 1, second - 1): value for first, second, value in STRESS}
    assert set(stress_map) <= set(pairs)
    stress = sp.Matrix([stress_map.get(pair, 0) for pair in pairs])
    assert stress.T * matrix == sp.zeros(1, 36)
    quadratic = []
    for first, second in pairs:
        difference = sp.Matrix(VELOCITIES[first]) - sp.Matrix(VELOCITIES[second])
        quadratic.append((difference.T * metric * difference)[0])
    energy = (stress.T * sp.Matrix(quadratic))[0]
    assert energy == 32
    known = {(row, column) for row in range(1, 5) for column in range(1, 5)}
    for location, centers in EXTENSION:
        assert location not in known and all(center in known for center in centers)
        assert all(tuple(sorted((location, center))) in full_pairs for center in centers)
        directions = [points[center] - points[centers[0]] for center in centers[1:]]
        assert directions[0].cross(directions[1]) != sp.zeros(3, 1)
        known.add(location)
    assert known == set(points)
    return {
        'scope': 'The precise initial configuration and whole-edge topology in the supplied 5x5 nonplanar static file only',
        'static_sha256': hashlib.sha256(static_path.read_bytes()).hexdigest(),
        'scaled_coordinates': {f'{row},{column}': list(map(int, value)) for (row, column), value in points.items()},
        'metric': [1, 1, 2],
        'patch_coordinates': [list(map(int, value)) for value in patch],
        'patch_original_names': [SOURCE_LABELS.get((row, column), f'W_{{{row}{column}}}')
                                  for row in range(1, 5) for column in range(1, 5)],
        'patch_pairs_one_based': [[first + 1, second + 1] for first, second in pairs],
        'anchor_point_ids_one_based': [2, 3, 6, 7],
        'free_coordinate_ids_one_based': [index + 1 for index in free],
        'matrix_shape': [42, 36], 'rank': 35,
        'minor_excluded_rows_one_based': sorted(excluded_rows),
        'minor_excluded_columns_one_based': [36],
        'minor_determinant': str(determinant),
        'velocity': VELOCITIES, 'stress': STRESS,
        'stress_times_matrix': '0', 'matrix_times_velocity': '0',
        'stress_energy': str(energy), 'extension_steps': EXTENSION,
        'conclusion': 'The anchored 3x3 subframework is isolated; three-sphere propagation fixes all 36 vertices. No noncongruent continuous rigid folding starts at this exact 5x5 configuration.',
    }


def export_full_construction(source, output, original):
    assert output.resolve() not in (source.resolve(), original.resolve())
    with ZipFile(source) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(contents['geogebra.xml'])
    before = definitions(root)
    original_definitions = definitions(read_xml(original))
    assert before[:len(original_definitions)] == original_definitions
    root.find('construction').set('title', '5x5 nonplanar - full construction - deformable linkage, not a rigid fold')
    elements = root.findall('construction/element')
    for element in elements:
        auxiliary = element.find('auxiliary')
        if auxiliary is None:
            auxiliary = ET.SubElement(element, 'auxiliary')
        auxiliary.set('val', 'false')
    root.find('kernel/algebraStyle').set('val', '3')
    root.find('kernel/decimals').set('val', '6')
    root.find('algebraView/mode').set('val', '3')
    for view in root.findall('gui/perspectives/perspective/views/view'):
        view.set('visible', str(view.get('id') in ('2', '512')).lower())
    for node in root.findall('gui/perspectives/perspective/input'):
        node.set('show', 'true')
    assert definitions(root) == before
    counts = dict(Counter(element.get('type') for element in elements))
    contents['geogebra.xml'] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    with ZipFile(output) as archive:
        assert archive.testzip() is None
    return {'output_file': str(output.relative_to(PROJECT)), 'element_counts': counts,
            'total_elements': len(elements), 'original_definitions_preserved': len(original_definitions),
            'all_definitions_preserved': len(before), 'geometry_changed': False,
            'all_elements_non_auxiliary': True, 'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-full-construction', action='store_true')
    arguments = parser.parse_args()
    report = verify(PROJECT / 'geogebra/5x5 nonplanar static.ggb')
    if arguments.export_full_construction:
        report['full_construction'] = export_full_construction(
            PROJECT / 'geogebra/5x5 nonplanar folding.ggb',
            PROJECT / 'geogebra/5x5 nonplanar folding full construction.ggb',
            PROJECT / 'geogebra/2x2 nonplanar folding.ggb')
    output = Path(__file__).with_name('5x5 nonplanar rigidity certificate.json')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('PASS: rank 35; exact minor = -2^36; Rv = 0; stress*R = 0; stress energy = 32.')
    print('PASS: all 20 outer vertices fixed by three noncollinear fixed centers.')
    if 'full_construction' in report:
        print(json.dumps(report['full_construction'], ensure_ascii=False))
    print(output)


if __name__ == '__main__':
    main()
