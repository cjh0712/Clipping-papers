"""Audit the approved patch with one position per vertex and independent intersections."""

import argparse
from collections import Counter
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
from scipy.optimize import least_squares

from extend_geogebra_grid import coordinates
from verify_geogebra_skew_source import source_intersections, validate_source


SOURCE_GRID = {(0, 0): 'O', (0, 1): 'D', (0, 2): 'C',
               (1, 0): 'F', (1, 1): 'A', (1, 2): 'B',
               (2, 0): 'K', (2, 1): 'G', (2, 2): 'Q'}


def read_construction(path):
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read('geogebra.xml'))
    points = {element.get('label'): coordinates(element)
              for element in root.findall('construction/element')
              if element.get('type') in ('point', 'point3d')}
    return root, points


def three_spheres(centers, radii):
    centers, radii = np.asarray(centers), np.asarray(radii)
    offsets = centers[1:] - centers[0]
    right_side = (radii[0] ** 2 - radii[1:] ** 2 + np.sum(offsets ** 2, axis=1)) / 2
    middle = offsets.T @ np.linalg.solve(offsets @ offsets.T, right_side)
    normal = np.cross(offsets[0], offsets[1])
    normal /= np.linalg.norm(normal)
    height_squared = radii[0] ** 2 - middle @ middle
    if height_squared < -1e-12:
        raise ValueError('No real three-sphere intersection.')
    candidates = centers[0] + middle + np.array([-1, 1])[:, None] * np.sqrt(max(0, height_squared)) * normal
    actual = np.linalg.norm(candidates[:, None] - centers[None, :], axis=-1)
    np.testing.assert_allclose(actual, np.broadcast_to(radii, actual.shape), atol=1e-11, rtol=0)
    return candidates


def reference_patch(points):
    labels = [[SOURCE_GRID.get((row, column), f'V_{{{row + 1}{column + 1}}}')
               for column in range(4)] for row in range(4)]
    vertices = np.array([[points[label] for label in row] for row in labels])
    faces = [[4 * row + column, 4 * row + column + 1,
              4 * (row + 1) + column + 1, 4 * (row + 1) + column]
             for row in range(3) for column in range(3)]
    incidence = Counter(tuple(sorted((first, second)))
                        for face in faces for first, second in zip(face, face[1:] + face[:1]))
    assert len(incidence) == 24 and sum(value == 2 for value in incidence.values()) == 12
    pairs = sorted({tuple(sorted(pair)) for face in faces for pair in combinations(face, 2)})
    return vertices, np.asarray(pairs), labels


def branch_candidates(source, reference):
    def radii(location, centers):
        return [np.linalg.norm(reference[location] - reference[center]) for center in centers]

    right_radii = radii((1, 3), ((1, 2), (0, 2), (2, 2)))
    lower_radii = radii((3, 1), ((2, 1), (2, 0), (2, 2)))
    horizontal_radii = radii((2, 3), ((1, 2), (1, 3), (2, 2)))
    vertical_radii = radii((3, 2), ((2, 1), (3, 1), (2, 2)))
    rights = three_spheres([source['B'], source['C'], source['Q']], right_radii)
    lowers = three_spheres([source['G'], source['K'], source['Q']], lower_radii)
    candidates = {}
    for right_index, lower_index in product(range(2), repeat=2):
        right, lower = rights[right_index], lowers[lower_index]
        horizontal = three_spheres([source['B'], right, source['Q']], horizontal_radii)
        vertical = three_spheres([source['G'], lower, source['Q']], vertical_radii)
        for horizontal_index, vertical_index in product(range(2), repeat=2):
            key = (right_index, lower_index, horizontal_index, vertical_index)
            candidates[key] = np.array([right, lower, horizontal[horizontal_index], vertical[vertical_index]])
    return candidates


def shared_vertex_fit(source, reference, pairs):
    current = reference.copy()
    for location, label in SOURCE_GRID.items():
        current[location] = source[label]
    fixed = {4 * row + column for row, column in SOURCE_GRID}
    free = [index for index in range(16) if index not in fixed]
    flat_reference = reference.reshape(16, 3)
    expected = np.linalg.norm(flat_reference[pairs[:, 0]] - flat_reference[pairs[:, 1]], axis=1)
    flat_current = current.reshape(16, 3)

    def residual(values):
        trial = flat_current.copy()
        trial[free] = values.reshape(-1, 3)
        return np.linalg.norm(trial[pairs[:, 0]] - trial[pairs[:, 1]], axis=1) - expected

    solution = least_squares(residual, flat_current[free].ravel(),
                             xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=2000)
    final_residual = residual(solution.x)
    worst = int(np.argmax(np.abs(final_residual)))
    return {'converged': bool(solution.success),
            'maximum_distance_error': float(np.max(np.abs(final_residual))),
            'distance_error_l2': float(np.linalg.norm(final_residual)),
            'worst_pair_vertex_indices': pairs[worst].tolist(),
            'interpretation': 'Residual diagnostic only; an approximate fit is not a rigid motion.'}


def audit(source_path, static_path):
    source_root, source_saved = read_construction(source_path)
    source_validation = validate_source(source_root)
    _, static_points = read_construction(static_path)
    reference, pairs, labels = reference_patch(static_points)
    angle = np.arctan2(static_points['F'][2], static_points['F'][0])
    source_initial = source_intersections(angle, source_saved)
    for location, label in SOURCE_GRID.items():
        np.testing.assert_allclose(reference[location], source_initial[label], atol=1e-12, rtol=0)
    candidates = branch_candidates(source_initial, reference)
    target = reference[[1, 3, 2, 3], [3, 1, 3, 2]]
    initial_errors = {key: float(np.max(np.linalg.norm(value - target, axis=1)))
                      for key, value in candidates.items()}
    selected = min(initial_errors, key=initial_errors.get)
    assert initial_errors[selected] < 1e-12
    assert sum(error < 1e-12 for error in initial_errors.values()) == 1
    expected_diagonal = float(np.linalg.norm(reference[2, 3] - reference[3, 2]))
    samples = []
    for offset in (-5, -1, -0.1, 0, 0.1, 1, 5):
        source = source_intersections(angle + np.radians(offset), source_initial)
        choices = branch_candidates(source, reference)
        lengths = {key: float(np.linalg.norm(points[2] - points[3])) for key, points in choices.items()}
        values = choices[selected]
        fitted = shared_vertex_fit(source, reference, pairs)
        if offset:
            assert abs(lengths[selected] - expected_diagonal) > 1e-6
            assert min(abs(length - expected_diagonal) for length in lengths.values()) > 1e-6
            assert fitted['maximum_distance_error'] > 1e-9
        else:
            assert abs(lengths[selected] - expected_diagonal) < 1e-12
            assert fitted['maximum_distance_error'] < 1e-12
        samples.append({
            'circle_offset_degrees': offset,
            'selected_branch_XY': lengths[selected],
            'required_XY': expected_diagonal,
            'selected_branch_YQ': float(np.linalg.norm(values[3] - source['Q'])),
            'selected_branch_Y_lower_neighbor': float(np.linalg.norm(values[3] - values[1])),
            'minimum_XY_error_among_16_branches': min(abs(length - expected_diagonal) for length in lengths.values()),
            'branch_XY_lengths': {str(key): value for key, value in lengths.items()},
            'single_shared_vertex_solver': fitted,
        })
    return {
        'status': 'no_exact_local_motion_found_for_the_stored_approved_geometry',
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'static_sha256': hashlib.sha256(static_path.read_bytes()).hexdigest(),
        'source_2x2_validation': source_validation,
        'method': 'Independent three-sphere intersections, with radii read from the approved static GGB; '
                  'no outside_neighbor, opposite_corner, eight_panel_patch, or blueJoin evaluation.',
        'topology': {'faces': 9, 'unique_vertices': 16, 'whole_edge_hinges': 12,
                     'one_position_per_shared_vertex': True, 'distance_constraints': len(pairs)},
        'vertex_labels': labels,
        'initial_branch_zero_based': list(selected),
        'initial_geometry_error': initial_errors[selected],
        'smallest_other_branch_geometry_change': min(value for key, value in initial_errors.items() if key != selected),
        'branches_checked_per_sample': 16,
        'Y_definition_check': 'Y is the intersection of three spheres centered at G, the lower neighbor, and Q. '
                              'Its adjacent yellow-panel edges remain length 2. The blue ninth panel requires '
                              'the additional distance XY = 2, not merely reusing the label Y.',
        'scope': 'This tests the coordinates in the stored static GGB, the source motion, and the stated rigid-panel '
                 'constraints. It is not a general claim that a skew origami can have at most eight faces.',
        'samples': samples,
    }


def write_report(report, output):
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    rows = ['| F 沿原圆转角 | 初态对应分支的 XY | 16 个交点组合中最小 XY 误差 | 共享顶点整体拟合的最大长度误差 |',
            '| ---: | ---: | ---: | ---: |']
    for sample in report['samples']:
        rows.append(f"| {sample['circle_offset_degrees']:+g}° | {sample['selected_branch_XY']:.9f} | "
                    f"{sample['minimum_XY_error_among_16_branches']:.9g} | "
                    f"{sample['single_shared_vertex_solver']['maximum_distance_error']:.9g} |")
    content = '\n'.join([
        '# Y 与整边连接的独立核查', '',
        '本次没有修改原始 2×2、已确认的静态 5×5 或 GeoGebra 中的运动定义。', '',
        '## 核查方法', '',
        '- 直接读取已确认静态 GGB 的坐标和各点间距离，不用上一版外围延拓公式。',
        '- 原四面由原文件的圆/平面交点独立计算，F 仍沿原圆运动，保留 K。',
        '- 外围点分别作为三个球面的交点求解；枚举影响 X、Y 的全部 16 种交点组合。',
        '- 另以 16 个唯一顶点、9 个面和 12 条整边接合整体拟合，不复制 Y，不允许裂缝。', '',
        '## 结果', '',
        'Y 与相邻黄色面的边始终连接，边长保持 2；Y 本身没有断开。',
        '上一版蓝色试接面却使用 blueJoin，因此它不是整边闭合的九面运动，不能作为完成结果。',
        '换成真正共用 Y 后，蓝色面的短对角线必须始终满足 XY = 2。',
        '对当前保存的静态坐标，独立计算仍未满足这一条件；整体共享顶点拟合也残留长度误差。', '',
        *rows, '',
        '初态对应的交点组合唯一。换用其他组合会改变已确认的初态，不是同一初态的另一套定义。',
        '这些计算检查的是当前保存的几何和刚性条件，不等于声称任意空间四边形模型都只能扩展八面。',
        '未输出带裂缝、面片拉伸或内部三角形相对转动的所谓折叠成果。', '',
    ])
    output.with_suffix('.md').write_text(content, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).parent
    parser.add_argument('source', type=Path)
    parser.add_argument('--static', type=Path, default=directory / 'geogebra' / '5x5 nonplanar static.ggb')
    parser.add_argument('--output', type=Path, default=directory / 'origami_skew_shared_vertex_audit.json')
    arguments = parser.parse_args()
    result = audit(arguments.source, arguments.static)
    write_report(result, arguments.output)
    print(json.dumps({key: value for key, value in result.items() if key != 'samples'}, ensure_ascii=False, indent=2))
    for sample in result['samples']:
        print(sample['circle_offset_degrees'], sample['selected_branch_XY'],
              sample['single_shared_vertex_solver']['maximum_distance_error'])
    print(arguments.output.with_suffix('.md').resolve())
