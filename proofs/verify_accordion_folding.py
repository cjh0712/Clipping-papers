"""Reproducible sampled surface-intersection check for the exact accordion."""

import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_geogebra_accordion_folding5x5 as model

TRIANGLE_PAIRS = np.array(list(itertools.combinations(range(50), 2)))


def geometry(angles):
    return (np.array([model.geometry(np.degrees(angle)).reshape(-1, 3) for angle in angles]),)


def projected_inside(point, triangle, tolerance):
    first_edge = triangle[1] - triangle[0]
    second_edge = triangle[2] - triangle[0]
    displacement = point - triangle[0]
    first_length = np.dot(first_edge, first_edge)
    mixed = np.dot(first_edge, second_edge)
    second_length = np.dot(second_edge, second_edge)
    first_dot = np.dot(displacement, first_edge)
    second_dot = np.dot(displacement, second_edge)
    denominator = first_length * second_length - mixed * mixed
    first_weight = (second_length * first_dot - mixed * second_dot) / denominator
    second_weight = (first_length * second_dot - mixed * first_dot) / denominator
    return first_weight >= -tolerance and second_weight >= -tolerance and first_weight + second_weight <= 1 + tolerance


def coplanar_contacts(first_triangle, second_triangle, tolerance):
    normal = np.cross(first_triangle[1] - first_triangle[0], first_triangle[2] - first_triangle[0])
    drop_axis = np.argmax(np.abs(normal))
    retained = [axis for axis in range(3) if axis != drop_axis]
    first_flat, second_flat = first_triangle[:, retained], second_triangle[:, retained]
    contacts = []
    for point in first_triangle:
        if projected_inside(point, second_triangle, tolerance):
            contacts.append(point)
    for point in second_triangle:
        if projected_inside(point, first_triangle, tolerance):
            contacts.append(point)
    for first_index in range(3):
        first_start = first_flat[first_index]
        first_delta = first_flat[(first_index + 1) % 3] - first_start
        for second_index in range(3):
            second_start = second_flat[second_index]
            second_delta = second_flat[(second_index + 1) % 3] - second_start
            matrix = np.stack((first_delta, -second_delta), axis=1)
            if abs(np.linalg.det(matrix)) <= tolerance:
                continue
            first_parameter, second_parameter = np.linalg.solve(matrix, second_start - first_start)
            if -tolerance <= first_parameter <= 1 + tolerance and -tolerance <= second_parameter <= 1 + tolerance:
                contacts.append(first_triangle[first_index] + first_parameter *
                                (first_triangle[(first_index + 1) % 3] - first_triangle[first_index]))
    return contacts


def triangle_contacts(first_triangle, second_triangle, tolerance):
    first_normal = np.cross(first_triangle[1] - first_triangle[0], first_triangle[2] - first_triangle[0])
    second_normal = np.cross(second_triangle[1] - second_triangle[0], second_triangle[2] - second_triangle[0])
    first_normal /= np.linalg.norm(first_normal)
    second_normal /= np.linalg.norm(second_normal)
    first_distances = (first_triangle - second_triangle[0]) @ second_normal
    second_distances = (second_triangle - first_triangle[0]) @ first_normal
    if np.max(np.abs(first_distances)) <= tolerance and np.max(np.abs(second_distances)) <= tolerance:
        return coplanar_contacts(first_triangle, second_triangle, tolerance), True
    contacts = []
    for source, target, distances in ((first_triangle, second_triangle, first_distances),
                                       (second_triangle, first_triangle, second_distances)):
        for edge_index in range(3):
            next_index = (edge_index + 1) % 3
            if abs(distances[edge_index]) <= tolerance and projected_inside(source[edge_index], target, tolerance):
                contacts.append(source[edge_index])
            denominator = distances[edge_index] - distances[next_index]
            if abs(denominator) <= tolerance:
                continue
            parameter = distances[edge_index] / denominator
            if -tolerance <= parameter <= 1 + tolerance:
                point = source[edge_index] + parameter * (source[next_index] - source[edge_index])
                if projected_inside(point, target, tolerance):
                    contacts.append(point)
    return contacts, False


def outside_shared_simplex(point, shared_points, tolerance):
    if len(shared_points) == 0:
        return True
    if len(shared_points) == 1:
        return np.linalg.norm(point - shared_points[0]) > tolerance
    edge = shared_points[1] - shared_points[0]
    parameter = np.clip(np.dot(point - shared_points[0], edge) / np.dot(edge, edge), 0, 1)
    return np.linalg.norm(point - shared_points[0] - parameter * edge) > tolerance


def collision_scan(angles, triangle_ids, tolerance, batch_size):
    first_ids, second_ids = TRIANGLE_PAIRS.T
    shared = [np.intersect1d(triangle_ids[first], triangle_ids[second]) for first, second in TRIANGLE_PAIRS]
    shared_counts = np.array([len(vertices) for vertices in shared])
    first_other = np.array([~np.isin(triangle_ids[first], common)
                            for (first, second), common in zip(TRIANGLE_PAIRS, shared)])
    second_other = np.array([~np.isin(triangle_ids[second], common)
                             for (first, second), common in zip(TRIANGLE_PAIRS, shared)])
    summary = {"configurations": len(angles), "pair_tests": int(len(angles) * len(TRIANGLE_PAIRS)),
               "aabb_candidates": 0, "plane_candidates": 0, "narrow_checks": 0,
               "coplanar_checks": 0, "collision_pairs": {}, "first_collision": None,
               "minimum_triangle_double_area": float("inf")}
    for start in range(0, len(angles), batch_size):
        points = geometry(angles[start:start + batch_size])[0]
        triangles = points[:, triangle_ids]
        normals = np.cross(triangles[:, :, 1] - triangles[:, :, 0], triangles[:, :, 2] - triangles[:, :, 0])
        normal_lengths = np.linalg.norm(normals, axis=-1)
        summary["minimum_triangle_double_area"] = min(summary["minimum_triangle_double_area"], float(normal_lengths.min()))
        normals /= normal_lengths[..., None]
        lower, upper = triangles.min(axis=2), triangles.max(axis=2)
        overlap = np.all((lower[:, first_ids] <= upper[:, second_ids] + tolerance)
                         & (lower[:, second_ids] <= upper[:, first_ids] + tolerance), axis=-1)
        summary["aabb_candidates"] += int(overlap.sum())
        sample_ids, pair_ids = np.nonzero(overlap)
        if not len(pair_ids):
            continue
        first_triangles = triangles[sample_ids, first_ids[pair_ids]]
        second_triangles = triangles[sample_ids, second_ids[pair_ids]]
        first_normals = normals[sample_ids, first_ids[pair_ids]]
        second_normals = normals[sample_ids, second_ids[pair_ids]]
        first_distances = np.sum((first_triangles - second_triangles[:, :1]) * second_normals[:, None], axis=-1)
        second_distances = np.sum((second_triangles - first_triangles[:, :1]) * first_normals[:, None], axis=-1)
        candidate = ((first_distances.min(axis=1) <= tolerance) & (first_distances.max(axis=1) >= -tolerance)
                     & (second_distances.min(axis=1) <= tolerance) & (second_distances.max(axis=1) >= -tolerance))
        summary["plane_candidates"] += int(candidate.sum())
        coplanar_pairs = ((np.max(np.abs(first_distances), axis=1) <= tolerance)
                          & (np.max(np.abs(second_distances), axis=1) <= tolerance))
        counts = shared_counts[pair_ids]
        candidate &= (counts != 2) | coplanar_pairs
        first_positive = np.min(np.where(first_other[pair_ids], first_distances, np.inf), axis=1) > tolerance
        first_negative = np.max(np.where(first_other[pair_ids], first_distances, -np.inf), axis=1) < -tolerance
        second_positive = np.min(np.where(second_other[pair_ids], second_distances, np.inf), axis=1) > tolerance
        second_negative = np.max(np.where(second_other[pair_ids], second_distances, -np.inf), axis=1) < -tolerance
        candidate &= ((counts != 1) | coplanar_pairs
                      | ~(first_positive | first_negative | second_positive | second_negative))
        for candidate_index in np.flatnonzero(candidate):
            pair_index = pair_ids[candidate_index]
            sample_index = sample_ids[candidate_index]
            common_vertices = shared[pair_index]
            first_triangle = first_triangles[candidate_index]
            second_triangle = second_triangles[candidate_index]
            summary["narrow_checks"] += 1
            contacts, was_coplanar = triangle_contacts(first_triangle, second_triangle, tolerance)
            summary["coplanar_checks"] += int(was_coplanar)
            unexpected = [point for point in contacts
                          if outside_shared_simplex(point, points[sample_index, common_vertices], tolerance * 20)]
            if unexpected:
                first_id, second_id = TRIANGLE_PAIRS[pair_index]
                key = f"{first_id},{second_id}"
                summary["collision_pairs"][key] = summary["collision_pairs"].get(key, 0) + 1
                if summary["first_collision"] is None:
                    summary["first_collision"] = {"degrees": float(np.degrees(angles[start + sample_index])),
                                                   "triangle_ids": [int(first_id), int(second_id)],
                                                   "vertices": [triangle_ids[first_id].tolist(), triangle_ids[second_id].tolist()],
                                                   "point": unexpected[0].tolist()}
        if start % (batch_size * 20) == 0:
            print(f"collision progress: {min(start + batch_size, len(angles))}/{len(angles)}", flush=True)
    return summary



def main():
    report = model.validate(samples=2401)
    report['surface_intersection_scan'] = collision_scan(
        np.radians(np.linspace(model.MIN_FOLD, model.MAX_FOLD, 2401)), model.TRIANGLE_IDS, 1e-10, 64)
    report['intersection_check_scope'] = '2401 sampled configurations; not an interval proof of collision freedom.'
    assert not report['surface_intersection_scan']['collision_pairs']
    destination = model.PROJECT / 'proofs' / (model.STEM + ' validation.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

