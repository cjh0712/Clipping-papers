"""Withdrawn J-branch experiment; not the user's original assembly."""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image

from extend_geogebra_grid import arc_points, coordinates


@dataclass
class Candidate:
    vertices: np.ndarray
    labels: np.ndarray
    recipes: list

    def transpose(self):
        return Candidate(self.vertices.transpose(1, 0, 2), self.labels.T, self.recipes)


def cone(first, second, dots, radius_squared):
    axes = np.array([first, second])
    middle = axes.T @ np.linalg.solve(axes @ axes.T, dots)
    normal = np.cross(first, second)
    normal /= np.linalg.norm(normal)
    remainder = radius_squared - middle @ middle
    if remainder < -1e-9:
        return []
    height = np.sqrt(max(0, remainder))
    return [middle + sign * height * normal for sign in (1, -1)]


def complete(origin, first, second, long):
    square = 8 if long else 4
    return [origin + direction for direction in
            cone(first-origin, second-origin, [square/2, square/2], square)]


def distinct(vertices):
    points = vertices.reshape(-1, 3)
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    return np.min(distances + np.eye(len(points))*100) > 1e-6


def extend_column(candidate, short=True):
    old = candidate.vertices[:, -1]
    labels = candidate.labels[:, -1]
    first_dot, second_dot = (2, 0) if short else (0, 2)
    new_labels = [f'W_{{{len(candidate.recipes) + index}}}' for index in range(len(old))]
    middle_label, top_label = new_labels[:2]
    new_labels[0], new_labels[1] = top_label, middle_label
    for middle_index, direction in enumerate(cone(old[0]-old[1], old[2]-old[1],
                                                   [first_dot, second_dot], 4)):
        middle = old[1]+direction
        incoming = candidate.vertices[1, -2]-old[1]
        difference = old[0]-old[2]
        reflected = incoming-2*np.dot(incoming, difference)/np.dot(difference, difference)*difference
        mode = 1 if np.linalg.norm(reflected-direction) < 1e-8 else -1
        middle_recipe = (middle_label, 'star', labels[1], labels[0], labels[2],
                         mode, short, candidate.labels[1, -2])
        for top_index, top in enumerate(complete(old[1], old[0], middle, short)):
            top_recipe = (top_label, 'quad', labels[1], labels[0], middle_label,
                          1-2*top_index, short, '')
            paths = [([top, middle], [middle_recipe, top_recipe])]
            for index in range(1, len(old)-1):
                next_paths = []
                for path, recipes in paths:
                    if abs((old[index+1]-old[index]) @ (path[-1]-old[index]) - second_dot) > 1e-8:
                        continue
                    for bottom_index, bottom in enumerate(complete(old[index], old[index+1],
                                                                   path[-1], not short)):
                        recipe = (new_labels[index+1], 'quad', labels[index], labels[index+1],
                                  new_labels[index], 1-2*bottom_index, not short, '')
                        next_paths.append(([*path, bottom], [*recipes, recipe]))
                paths = next_paths
            for path, recipes in paths:
                vertices = np.concatenate([candidate.vertices, np.array(path)[:, None]], axis=1)
                if distinct(vertices):
                    yield Candidate(vertices, np.column_stack([candidate.labels, new_labels]),
                                    [*candidate.recipes, *recipes])


def grow(points, corner='J', size=5):
    labels = np.rot90(np.array([['O', 'D', 'C'], ['F', 'A', 'B'], [corner, 'G', 'Q']]))
    vertices = np.array([[points[label] for label in row] for row in labels])
    candidates = [Candidate(vertices, labels, [])]
    counts = []
    for current_size in range(3, size+1):
        candidates = [grown.transpose() for candidate in candidates
                      for widened in extend_column(candidate)
                      for grown in extend_column(widened.transpose())]
        counts.append(len(candidates))
        if not candidates:
            break
    return candidates, counts


def smoothness(candidate):
    vertices = candidate.vertices
    return sum(float(np.sum(np.diff(vertices, n=2, axis=axis)**2)) for axis in (0, 1))


def evaluate(points, recipes):
    result = dict(points)
    for label, kind, origin_label, first_label, second_label, sign, flag, incoming_label in recipes:
        origin, first, second = [result[name] for name in (origin_label, first_label, second_label)]
        if kind == 'quad':
            choices = complete(origin, first, second, flag)
        else:
            incoming = result[incoming_label]-origin
            axis = first-second if sign == 1 else first+second-2*origin
            direction = sign*(incoming-2*np.dot(incoming, axis)/np.dot(axis, axis)*axis)
            result[label] = origin+direction
            continue
        if not choices:
            raise ValueError(f'No real construction for {label}')
        result[label] = choices[0 if sign == 1 else 1]
    return result


def source_points(angle):
    points = arc_points(angle)
    first, second, driven, follower = [points[name] for name in ('B', 'D', 'F', 'G')]
    points.update(C=np.array([1, 1, -np.sqrt(2)]),
                  O=2/3*(second+driven)-np.sqrt(2)/3*np.cross(second, driven),
                  Q=2/3*(first+follower)+np.sqrt(2)/3*np.cross(first, follower),
                  J=(driven+follower)/2+np.cross(driven, follower)/(2*np.sqrt(2)))
    return points


def face_vertices(vertices):
    return np.stack([vertices[:-1, :-1], vertices[:-1, 1:],
                     vertices[1:, 1:], vertices[1:, :-1]], axis=-2).reshape(-1, 4, 3)


def validate(candidate, saved):
    for label in ('A', 'B', 'C', 'D', 'F', 'G', 'J', 'O', 'Q'):
        np.testing.assert_allclose(source_points(np.arctan2(saved['F'][2], saved['F'][0]))[label],
                                   saved[label], atol=1e-10)
    reference = face_vertices(candidate.vertices)
    target = np.array([[0, 2, 2, 2], [2, 0, 2, np.sqrt(8)],
                       [2, 2, 0, 2], [2, np.sqrt(8), 2, 0]])
    np.testing.assert_allclose(np.linalg.norm(reference[:, :, None]-reference[:, None, :], axis=-1),
                               np.broadcast_to(target, (25, 4, 4)), atol=1e-9)
    worst = 0.0
    corner_signs = []
    edge_incidence = {}
    for row in range(5):
        for column in range(5):
            labels = [candidate.labels[location] for location in
                      ((row, column), (row, column+1), (row+1, column+1), (row+1, column))]
            for first, second in zip(labels, labels[1:]+labels[:1]):
                key = tuple(sorted((first, second)))
                edge_incidence[key] = edge_incidence.get(key, 0)+1
    assert len(set(candidate.labels.ravel())) == 36
    assert len(edge_incidence) == 60 and list(edge_incidence.values()).count(2) == 40
    for degrees in np.linspace(1, 179, 721):
        points = evaluate(source_points(np.radians(degrees)), candidate.recipes)
        vertices = np.array([[points[label] for label in row] for row in candidate.labels])
        faces = face_vertices(vertices)
        center = vertices[1:-1, 1:-1]
        up, left, down = vertices[:-2, 1:-1]-center, vertices[1:-1, :-2]-center, vertices[2:, 1:-1]-center
        corner_signs.append(np.sum(np.cross(up, left)*down, axis=-1))
        error = float(np.max(np.abs(np.linalg.norm(faces[:, :, None]-faces[:, None, :], axis=-1)-target)))
        worst = max(worst, error)
    if worst > 1e-7:
        raise AssertionError(f'Rigid-panel error across circle: {worst}')
    corner_signs = np.array(corner_signs)
    assert not np.any((corner_signs.min(axis=0) < -1e-8) & (corner_signs.max(axis=0) > 1e-8))
    print(f'PASS: 721 samples on the source upper-circle branch, max six-distance error {worst:.3e}; '
          '36 vertices, 25 rigid skew quads, 40 shared whole edges, 16 unchanged vertex signs.')
    return worst


def expression(recipe):
    label, kind, origin, first, second, sign, flag, incoming_label = recipe
    first_vector = f'Vector({origin},{first})'
    second_vector = f'Vector({origin},{second})'
    normal = f'Cross({first_vector},{second_vector})'
    if kind == 'quad':
        middle = f'(2/3)*({first_vector}+{second_vector})' if flag else f'({first_vector}+{second_vector})/2'
        height = 'sqrt(2)/3' if flag else '1/(2*sqrt(2))'
    else:
        incoming = f'Vector({origin},{incoming_label})'
        axis = f'({first_vector}{"-" if sign == 1 else "+"}{second_vector})'
        return f'{origin}+({sign})*({incoming}-2*({incoming}*{axis})/({axis}*{axis})*{axis})'
    return f'{origin}+({middle}){ "+" if sign == 1 else "-" }({height})*{normal}'


def preview(candidate, path):
    figure = plt.figure(figsize=(10, 8), facecolor='white')
    axis = figure.add_subplot(projection='3d')
    faces = face_vertices(candidate.vertices)
    triangles, lines, colors = [], [], []
    for index, face in enumerate(faces):
        is_yellow = (index//5+index%5)%2 == 0
        triangles.extend([face[[0, 1, 3]], face[[1, 2, 3]]] if is_yellow else
                         [face[[0, 1, 2]], face[[0, 2, 3]]])
        lines.append(face[[0, 1, 2, 3, 0]])
        colors.extend(['#e9b941' if is_yellow else '#2b83b4']*2)
    axis.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors='none', alpha=0.9))
    axis.add_collection3d(Line3DCollection(lines, colors='#263c52', linewidths=0.9))
    lower, upper = candidate.vertices.min(axis=(0, 1)), candidate.vertices.max(axis=(0, 1))
    axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
    axis.set_box_aspect(upper-lower)
    axis.set_proj_type('ortho')
    axis.view_init(elev=65, azim=-45)
    axis.set_axis_off()
    figure.suptitle('5 × 5 skew quadrilaterals · alternate J branch', fontsize=16)
    figure.text(0.5, 0.04, 'Candidate: one source panel uses J instead of K; original circle driver retained.',
                ha='center', fontsize=10, color='#46566a')
    figure.savefig(path, dpi=150, bbox_inches='tight')
    buffer = BytesIO()
    figure.savefig(buffer, format='png', dpi=40, bbox_inches='tight')
    plt.close(figure)
    thumbnail = Image.open(buffer)
    thumbnail.thumbnail((256, 256))
    packed = BytesIO()
    thumbnail.save(packed, format='PNG')
    return packed.getvalue()


def build(source, output):
    if source.resolve() == output.resolve():
        raise ValueError('Keep the supplied file unchanged; choose a separate output.')
    with ZipFile(source) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root = ET.fromstring(contents['geogebra.xml'])
    construction = root.find('construction')
    elements = {element.get('label'): element for element in construction.findall('element')}
    points = {label: coordinates(element) for label, element in elements.items()
              if element.get('type') in ('point', 'point3d')}
    original = [ET.tostring(node) for node in construction if node.tag in ('command', 'expression')]
    _, original_counts = grow(points, corner='K')
    candidates, candidate_counts = grow(points)
    print(f'Original K branch extension counts (3x3 onward): {original_counts}')
    print(f'Alternate J branch extension counts (3x3 onward): {candidate_counts}')
    if not candidates:
        raise ValueError('No non-coincident 5x5 candidate found.')
    candidate = min(candidates, key=smoothness)
    validate(candidate, points)
    for element in construction.findall('element'):
        show = element.find('show')
        if show is not None:
            visible = element.get('label') in ('e', 'F')
            show.set('object', str(visible).lower())
            show.set('label', str(element.get('label') == 'F').lower())
    construction.set('title', '5x5 skew candidate - alternate J branch, original circle driver')
    point_values = evaluate(points, candidate.recipes)
    for recipe in candidate.recipes:
        label = recipe[0]
        ET.SubElement(construction, 'expression', label=label,
                      exp=f'If(z(F)>0,{expression(recipe)})', type='point')
        element = deepcopy(elements['J'])
        element.set('label', label)
        element.find('coords').attrib.update(dict(zip(('x', 'y', 'z'), map(str, point_values[label]))))
        construction.append(element)
    for row in range(5):
        for column in range(5):
            quad = [candidate.labels[location] for location in
                    ((row, column), (row, column+1), (row+1, column+1), (row+1, column))]
            is_yellow = (row+column)%2 == 0
            triangles = ((0, 1, 3), (1, 2, 3)) if is_yellow else ((0, 1, 2), (0, 2, 3))
            for triangle_index, corner_ids in enumerate(triangles):
                face_label = f'S_{{{row}{column}{triangle_index}}}'
                edge_labels = [f'b_{{{row}{column}{triangle_index}{index}}}' for index in range(3)]
                command = ET.SubElement(construction, 'command', name='Polygon')
                ET.SubElement(command, 'input', {f'a{index}': quad[corner] for index, corner in enumerate(corner_ids)})
                ET.SubElement(command, 'output', {f'a{index}': label for index, label in enumerate([face_label, *edge_labels])})
                element = deepcopy(elements['t1' if is_yellow else 't5'])
                element.set('label', face_label)
                element.find('show').set('object', 'true')
                element.find('lineStyle').set('thickness', '0')
                construction.append(element)
                for edge_index, edge_label in enumerate(edge_labels):
                    edge = deepcopy(elements['d_1'])
                    edge.set('label', edge_label)
                    diagonal_edge = (1, 2)[triangle_index] if is_yellow else (2, 0)[triangle_index]
                    diagonal = edge_index == diagonal_edge
                    edge.find('show').set('object', str(not diagonal).lower())
                    edge.find('show').set('label', 'false')
                    edge.find('objColor').attrib.update(r='38', g='60', b='82')
                    edge.find('lineStyle').set('thickness', '3')
                    construction.append(edge)
    elements['F'].find('pointSize').set('val', '7')
    elements['F'].find('objColor').attrib.update(r='204', g='51', b='68')
    center = candidate.vertices.mean(axis=(0, 1))
    camera = root.find('euclidianView3D/coordSystem')
    camera.attrib.update({axis+'Zero': str(-value) for axis, value in zip(('x', 'y', 'z'), center)})
    camera.attrib.update(scale='43', xAngle='52', zAngle='-135')
    retained = [ET.tostring(node) for node in construction if node.tag in ('command', 'expression')][:len(original)]
    assert retained == original
    contents['geogebra.xml'] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    contents['geogebra_thumbnail.png'] = preview(candidate, output.with_suffix('.png'))
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    print(f'Candidate (not identical to source assembly): {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.error('The J-branch candidate has been withdrawn. Use build_geogebra_skew_grid5x5.py '
                 'for the original K-oriented static reference and its rigidity audit.')


if __name__ == '__main__':
    main()
