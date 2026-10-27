"""Quality metrics for grain boundary meshes and volume meshes (phase M0).

Every later meshing step (M1-M4) is judged with these functions, and so is the legacy mesher.
They take plain arrays, know nothing about how the mesh was made and are checked against the
staircase mesh, whose answers are known exactly (tests/unit/test_mesh_quality.py).

Surface checks, per grain, on the triangles of an InterfaceMesh oriented outward for that grain:
  open_edges          edges used by an odd number of the grain's triangles (hole in the surface)
  nonmanifold_edges   edges used by more than two of them (e.g. two voxels touching at an edge)
  misoriented_edges   closed edges whose two triangles run the same way along them
  nonmanifold_vertices vertices around which the grain's triangles form more than one fan
                       (e.g. two voxels touching at a corner)
  components / euler  connected pieces of the surface (connected through shared edges) and the
                       Euler characteristic V - E + F of each; 2 for a sphere, 0 for a torus
  volume              enclosed volume (divergence theorem); equals the voxel volume for the
                       staircase mesh

Interface checks: duplicate triangles, degenerate triangles, triangles with the same grain on
both sides, triangles facing a GAP (non-conforming neighbours), smallest angle and radius ratio,
pairwise self-intersections. For smoothed meshes (M3): kink angle between neighbouring triangles
of a patch, triangles folded against their staircase position.

Volume checks: inverted elements (det J <= 0 at a check point), scaled Jacobian, Jacobian ratio,
dihedral angles and radius ratio of tetrahedra, faces used once that do not lie on the box
(gaps between grains = non-conforming mesh), faces used more than twice, tet10 midside nodes that
differ between neighbours, element volume per grain, connected element bodies per grain.

Periodicity: every node on a lower box face has a partner at the same position on the upper face,
and the boundary facets on both faces match through these partners.
"""
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from dragen.mesh.surface import GAP, InterfaceMesh, is_box_face
from dragen.mesh.volume import (TET10_EDGES, VolumeMesh, element_volumes, jacobian_determinants,
                                scaled_jacobian, tet_dihedral_angles, tet_radius_ratio)


def _group(*cols) -> np.ndarray:
    """Group ID of every row of the given integer columns (rows with equal values share one)."""
    order = np.lexsort(cols[::-1])
    keys = np.stack([np.asarray(c)[order] for c in cols])
    start = np.concatenate([[True], np.any(keys[:, 1:] != keys[:, :-1], axis=0)])
    ids = np.empty(len(order), dtype=np.int64)
    ids[order] = np.cumsum(start) - 1
    return ids


def unique_pairs(a, b, return_inverse: bool = False, return_counts: bool = False):
    """np.unique over the rows (a, b) of two integer columns.

    Encodes each row in one int64 key and sorts once; several times faster than
    np.unique(..., axis=0), which sorts rows as structured records. Returns the unique rows
    (n, 2), then the inverse and the counts if asked for.
    """
    a = np.asarray(a, dtype=np.int64).ravel()
    b = np.asarray(b, dtype=np.int64).ravel()
    if not len(a):
        rows = np.zeros((0, 2), dtype=np.int64)
        extra = [np.zeros(0, dtype=np.int64)] * (return_inverse + return_counts)
        return (rows, *extra) if extra else rows
    a0, b0 = a.min(), b.min()
    span = int(b.max() - b0) + 1
    if (int(a.max() - a0) + 1) * span >= 2 ** 62:
        raise OverflowError('pair values too large to encode')
    out = np.unique((a - a0) * span + (b - b0), return_inverse=return_inverse, return_counts=return_counts)
    keys = out[0] if isinstance(out, tuple) else out
    rows = np.column_stack([keys // span + a0, keys % span + b0])
    if isinstance(out, tuple):
        return (rows, *[o.ravel() for o in out[1:]])
    return rows


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    graph = sparse.coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def _chain_pairs(group: np.ndarray):
    """Positions of consecutive rows within each group (after sorting by group)."""
    order = np.argsort(group, kind='stable')
    same = group[order[1:]] == group[order[:-1]]
    return order[:-1][same], order[1:][same]


# --------------------------------------------------------------------------------------------- #
# surfaces

def grain_surfaces(mesh: InterfaceMesh) -> pd.DataFrame:
    """One row per grain with the surface checks from the module docstring."""
    grain, tris, _ = mesh.sided()
    s = len(tris)
    t = np.repeat(np.arange(s), 3)
    k = np.tile(np.arange(3), s)
    u = tris.ravel()
    v = tris[:, [1, 2, 0]].ravel()
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    g = grain[t]
    edge = _group(g, lo, hi)
    n_edges = edge.max() + 1 if len(edge) else 0
    use = np.bincount(edge, minlength=n_edges)
    turn = np.bincount(edge, weights=np.where(u < v, 1, -1), minlength=n_edges)
    edge_grain = np.zeros(n_edges, dtype=np.int64)
    edge_grain[edge] = g

    # triangles connected through shared edges; corners connected around a vertex
    a, b = _chain_pairs(edge)
    component = _components(s, t[a], t[b])
    corner_lo = 3 * t + np.where(u < v, k, (k + 1) % 3)
    corner_hi = 3 * t + np.where(u < v, (k + 1) % 3, k)
    fan = _components(3 * s, np.concatenate([corner_lo[a], corner_hi[a]]),
                      np.concatenate([corner_lo[b], corner_hi[b]]))
    corner_vertex = tris.ravel()
    corner_grain = np.repeat(grain, 3)
    gv = _group(corner_grain, corner_vertex)
    fans_per_gv = np.bincount(unique_pairs(gv, fan)[:, 0])
    gv_grain = np.zeros(len(fans_per_gv), dtype=np.int64)
    gv_grain[gv] = corner_grain

    # Euler characteristic per connected piece
    n_comp = component.max() + 1 if s else 0
    faces = np.bincount(component, minlength=n_comp)
    edges = np.bincount(unique_pairs(component[t], edge)[:, 0], minlength=n_comp)
    verts = np.bincount(unique_pairs(np.repeat(component, 3), corner_vertex)[:, 0], minlength=n_comp)
    euler = verts - edges + faces
    comp_grain = np.zeros(n_comp, dtype=np.int64)
    comp_grain[component] = grain

    p = mesh.points[tris]
    volume = np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2])) / 6

    ids = np.unique(grain)
    pos = {gid: i for i, gid in enumerate(ids)}

    def per_grain(owner, values):
        out = np.zeros(len(ids), dtype=np.asarray(values).dtype)
        np.add.at(out, np.searchsorted(ids, owner), values)
        return out

    euler_lists = [[] for _ in ids]
    for c in np.argsort(euler, kind='stable'):
        euler_lists[pos[comp_grain[c]]].append(int(euler[c]))
    open_edges = per_grain(edge_grain, (use % 2 == 1).astype(np.int64))
    nonmanifold_edges = per_grain(edge_grain, (use > 2).astype(np.int64))
    misoriented = per_grain(edge_grain, ((use % 2 == 0) & (turn != 0)).astype(np.int64))
    nonmanifold_vertices = per_grain(gv_grain, (fans_per_gv > 1).astype(np.int64))
    return pd.DataFrame({
        'grain': ids,
        'triangles': np.bincount(np.searchsorted(ids, grain), minlength=len(ids)),
        'open_edges': open_edges,
        'nonmanifold_edges': nonmanifold_edges,
        'misoriented_edges': misoriented,
        'nonmanifold_vertices': nonmanifold_vertices,
        'components': per_grain(comp_grain, np.ones(n_comp, dtype=np.int64)),
        'euler': [tuple(e) for e in euler_lists],
        'closed_manifold': (open_edges + nonmanifold_edges + misoriented + nonmanifold_vertices) == 0,
        'volume': per_grain(grain, volume),
    })


def interface_patches(mesh: InterfaceMesh) -> Dict[tuple, int]:
    """Connected pieces of the boundary between every pair of grains (box faces left out).

    Triangles of the same pair are connected when they share an edge.
    """
    inner = ~is_box_face(mesh.grains).any(axis=1)
    tris = mesh.triangles[inner]
    pair = np.sort(mesh.grains[inner], axis=1)
    if not len(tris):
        return {}
    f = len(tris)
    t = np.repeat(np.arange(f), 3)
    u, v = tris.ravel(), tris[:, [1, 2, 0]].ravel()
    edge = _group(pair[t, 0], pair[t, 1], np.minimum(u, v), np.maximum(u, v))
    a, b = _chain_pairs(edge)
    piece = _components(f, t[a], t[b])
    keys, counts = np.unique(np.unique(np.column_stack([pair, piece]), axis=0)[:, :2], axis=0,
                             return_counts=True)
    return {(int(g1), int(g2)): int(n) for (g1, g2), n in zip(keys, counts)}


def surface_kinks(points: np.ndarray, triangles: np.ndarray, patch: np.ndarray) -> np.ndarray:
    """Angle in degrees between the normals of every two triangles that share an edge inside the
    same patch (0 = flat; the staircase has 0 and 90). Triangles must be consistently oriented
    within a patch."""
    t = points[triangles]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    tri = np.repeat(np.arange(len(triangles)), 3)
    u, v = triangles.ravel(), triangles[:, [1, 2, 0]].ravel()
    key = unique_pairs(np.minimum(u, v), np.maximum(u, v), return_inverse=True)[1]
    order = np.argsort(key, kind='stable')
    same = key[order[1:]] == key[order[:-1]]
    i, j = tri[order[:-1]][same], tri[order[1:]][same]
    keep = patch[i] == patch[j]
    cos = np.einsum('ij,ij->i', n[i[keep]], n[j[keep]])
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def folded_triangles(reference: np.ndarray, points: np.ndarray, triangles: np.ndarray) -> int:
    """Triangles whose normal turned by 90 degrees or more against the reference positions."""
    def normals(x):
        t = x[triangles]
        return np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    return int(np.count_nonzero(np.einsum('ij,ij->i', normals(points), normals(reference)) <= 0))


def triangle_quality(points: np.ndarray, triangles: np.ndarray) -> Dict[str, np.ndarray]:
    """Smallest angle (degrees), radius ratio 2 r_in / r_circ (1 = equilateral) and area per triangle."""
    p = points[triangles]
    e = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]], axis=1)
    length = np.linalg.norm(e, axis=2)
    area = np.linalg.norm(np.cross(e[:, 0], -e[:, 2]), axis=1) / 2
    angles = []
    for i in range(3):
        a, b = -e[:, i - 1], e[:, i]          # the two edges leaving vertex i
        cos = np.einsum('ij,ij->i', a, b) / np.maximum(length[:, i - 1] * length[:, i], 1e-300)
        angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    with np.errstate(divide='ignore', invalid='ignore'):
        r_in = 2 * area / length.sum(axis=1)
        r_circ = np.prod(length, axis=1) / (4 * area)
        ratio = np.nan_to_num(2 * r_in / r_circ)
    return {'min_angle': np.min(angles, axis=0), 'radius_ratio': ratio, 'area': area}


def self_intersections(points: np.ndarray, triangles: np.ndarray, tol: float = 1e-9) -> int:
    """Number of triangle pairs without a common vertex whose interiors cross.

    Broad phase: pairs whose centroids are closer than the sum of their bounding radii.
    Narrow phase: an edge of one triangle passes through the interior of the other
    (Moeller-Trumbore). Coplanar overlaps are not detected; exact duplicates are counted by
    `interface_report` instead.
    """
    return len(self_intersection_pairs(points, triangles, tol=tol))


def self_intersection_pairs(points: np.ndarray, triangles: np.ndarray, candidates: np.ndarray = None,
                            tol: float = 1e-9) -> np.ndarray:
    """The crossing pairs (k, 2) of `self_intersections`; with `candidates` (triangle indices)
    only pairs that involve at least one candidate are tested."""
    p = points[triangles]
    centre = p.mean(axis=1)
    radius = np.linalg.norm(p - centre[:, None], axis=2).max(axis=1)
    tree = cKDTree(centre)
    if candidates is None:
        pairs = tree.query_pairs(2 * radius.max(), output_type='ndarray')
    else:
        candidates = np.unique(np.asarray(candidates, dtype=np.int64))
        near = tree.query_ball_point(centre[candidates], 2 * radius.max())
        rows = np.repeat(candidates, [len(n) for n in near])
        cols = np.concatenate([np.asarray(n, dtype=np.int64) for n in near]) if len(near) else np.zeros(0, np.int64)
        pairs = np.unique(np.sort(np.column_stack([rows, cols]), axis=1), axis=0)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    i, j = pairs[:, 0], pairs[:, 1]
    near = np.linalg.norm(centre[i] - centre[j], axis=1) <= radius[i] + radius[j]
    i, j = i[near], j[near]
    shared = (triangles[i][:, :, None] == triangles[j][:, None, :]).any(axis=(1, 2))
    i, j = i[~shared], j[~shared]
    hit = np.zeros(len(i), dtype=bool)
    for a, b in ((i, j), (j, i)):
        for k in range(3):
            hit |= _segment_hits_triangle(p[a, k], p[a, (k + 1) % 3], p[b], tol)
    return np.column_stack([i[hit], j[hit]])


def _segment_hits_triangle(s0, s1, tri, tol):
    d = s1 - s0
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, h)
    ok = np.abs(det) > tol * np.linalg.norm(d, axis=1) * np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
    det = np.where(ok, det, 1.0)
    s = s0 - tri[:, 0]
    uu = np.einsum('ij,ij->i', s, h) / det
    q = np.cross(s, e1)
    vv = np.einsum('ij,ij->i', d, q) / det
    tt = np.einsum('ij,ij->i', e2, q) / det
    return ok & (uu > tol) & (vv > tol) & (uu + vv < 1 - tol) & (tt > tol) & (tt < 1 - tol)


def interface_report(mesh: InterfaceMesh, reference_volume: Optional[Dict[int, float]] = None,
                     intersections: bool = False) -> dict:
    """Summary of all surface checks; reference_volume maps grain ID -> expected volume."""
    grains = grain_surfaces(mesh)
    quality = triangle_quality(mesh.points, mesh.triangles)
    key = np.sort(mesh.triangles, axis=1)
    _, dup_counts = np.unique(key, axis=0, return_counts=True)
    scale = np.median(quality['area']) if len(quality['area']) else 1.0
    out = {
        'n_points': int(len(mesh.points)),
        'n_triangles': int(len(mesh.triangles)),
        'n_grains': int(len(grains)),
        'duplicate_triangles': int((dup_counts - 1).sum()),
        'degenerate_triangles': int(np.count_nonzero(quality['area'] < 1e-10 * scale)),
        'same_region_triangles': int(np.count_nonzero(mesh.grains[:, 0] == mesh.grains[:, 1])),
        'gap_triangles': int(np.count_nonzero(mesh.grains == GAP)),
        'grains_not_closed_manifold': int(np.count_nonzero(~grains['closed_manifold'])),
        'open_edges': int(grains['open_edges'].sum()),
        'nonmanifold_edges': int(grains['nonmanifold_edges'].sum()),
        'misoriented_edges': int(grains['misoriented_edges'].sum()),
        'nonmanifold_vertices': int(grains['nonmanifold_vertices'].sum()),
        'grains_with_several_surfaces': int(np.count_nonzero(grains['components'] > 1)),
        'min_angle_deg': _quantiles(quality['min_angle']),
        'radius_ratio': _quantiles(quality['radius_ratio']),
    }
    if reference_volume is not None:
        out['volume_deviation'] = _volume_deviation(grains['grain'].to_numpy(), grains['volume'].to_numpy(),
                                                    reference_volume)
    if intersections:
        out['self_intersections'] = self_intersections(mesh.points, mesh.triangles)
    return out


# --------------------------------------------------------------------------------------------- #
# volume meshes

def volume_report(mesh: VolumeMesh, box_lower: Sequence[float], box_upper: Sequence[float],
                  reference_volume: Optional[Dict[int, float]] = None, tol: float = 1e-9,
                  periodic_axes: Sequence[int] = ()) -> dict:
    """Summary of the volume checks from the module docstring. tol is relative to the box size.

    On periodic_axes, elements of the same grain on opposite box faces whose faces mirror each
    other belong to the same body.
    """
    det = jacobian_determinants(mesh)
    volume = element_volumes(mesh)
    sj = scaled_jacobian(mesh)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.nan_to_num(det.min(axis=1) / det.max(axis=1))
    faces, owner, use = _element_faces(mesh)
    once = use[faces[1]] == 1
    on_box = _on_box(mesh.points, faces[0], box_lower, box_upper, tol)
    out = {
        'cell_type': mesh.cell_type,
        'n_points': int(len(mesh.points)),
        'n_elements': int(len(mesh.cells)),
        'unused_points': int(len(mesh.points) - len(np.unique(mesh.cells))),
        'inverted_elements': int(np.count_nonzero(det.min(axis=1) <= 0)),
        'scaled_jacobian': _quantiles(sj),
        'jacobian_ratio': _quantiles(ratio),
        'open_inner_faces': int(np.count_nonzero(once & ~on_box)),
        'overused_faces': int(np.count_nonzero(np.bincount(faces[1]) > 2)),
        'grains_with_several_bodies': int(np.count_nonzero(
            _bodies_per_grain(mesh, faces, owner, box_lower, box_upper, periodic_axes, tol) > 1)),
    }
    if mesh.cell_type in ('tet4', 'tet10'):
        dihedral = tet_dihedral_angles(mesh)
        out['dihedral_min_deg'] = _quantiles(dihedral.min(axis=1))
        out['dihedral_max_deg'] = _quantiles(dihedral.max(axis=1), upper=True)
        out['radius_ratio'] = _quantiles(tet_radius_ratio(mesh))
    if mesh.cell_type == 'tet10':
        out['midside_mismatch'] = _midside_mismatch(mesh)
    if reference_volume is not None:
        ids = np.unique(mesh.grain)
        per_grain = np.bincount(np.searchsorted(ids, mesh.grain), weights=volume, minlength=len(ids))
        out['volume_deviation'] = _volume_deviation(ids, per_grain, reference_volume)
    return out


def _element_faces(mesh: VolumeMesh):
    """Faces as rows of sorted corner nodes, a face ID per (element, face) and use counts."""
    local = mesh.faces
    nodes = mesh.corners[:, local]                       # (E, f, k)
    rows = np.sort(nodes.reshape(-1, local.shape[1]), axis=1)
    face_id = _group(*rows.T)
    owner = np.repeat(np.arange(len(mesh.cells)), len(local))
    return (rows, face_id), owner, np.bincount(face_id)


def _on_box(points, rows, lower, upper, tol):
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    eps = tol * np.max(upper - lower)
    x = points[rows]                                     # (n, k, 3)
    flat = np.zeros(len(rows), dtype=bool)
    for bound in (lower, upper):
        flat |= np.any(np.all(np.abs(x - bound) <= eps, axis=1), axis=1)
    return flat


def _bodies_per_grain(mesh, faces, owner, lower, upper, periodic_axes, tol):
    rows, face_id = faces
    a, b = _chain_pairs(face_id)
    a, b = owner[a], owner[b]
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    eps = tol * np.max(upper - lower)
    once = np.bincount(face_id)[face_id] == 1
    centre = mesh.points[rows].mean(axis=1)
    for axis in periodic_axes:
        lo = np.flatnonzero(once & (np.abs(centre[:, axis] - lower[axis]) <= eps))
        hi = np.flatnonzero(once & (np.abs(centre[:, axis] - upper[axis]) <= eps))
        if not len(lo) or not len(hi):
            continue
        shifted = centre[lo].copy()
        shifted[:, axis] = upper[axis]
        dist, idx = cKDTree(centre[hi]).query(shifted, distance_upper_bound=2 * eps + 1e-300)
        found = np.isfinite(dist)
        a = np.concatenate([a, owner[lo[found]]])
        b = np.concatenate([b, owner[hi[idx[found]]]])
    same = mesh.grain[a] == mesh.grain[b]
    body = _components(len(mesh.cells), a[same], b[same])
    pairs = unique_pairs(mesh.grain, body)
    return np.unique(pairs[:, 0], return_counts=True)[1]


def _midside_mismatch(mesh):
    """Element edges that share both corners with another element edge but not the midside node."""
    corners = mesh.cells[:, TET10_EDGES]                 # (E, 6, 2)
    mid = mesh.cells[:, 4:].ravel()
    lo = corners.min(axis=2).ravel()
    hi = corners.max(axis=2).ravel()
    edge = _group(lo, hi)
    variants = unique_pairs(edge, mid)[:, 0]
    return int(np.count_nonzero(np.bincount(variants) > 1))


# --------------------------------------------------------------------------------------------- #
# periodicity and anchors

def periodic_mismatch(points: np.ndarray, facets: np.ndarray, box_lower: Sequence[float],
                      box_upper: Sequence[float], axes: Sequence[int], tol: float = 1e-9) -> dict:
    """Partner check between opposite box faces on the given axes.

    facets: boundary facets (triangles or element faces, any number of nodes); facets that do
    not lie on one of the two faces are ignored. tol is relative to the box size.
    Returns per axis the node counts on both faces, nodes without a partner, the largest
    distance to the partner, and facets without a mirrored facet.
    """
    lower, upper = np.asarray(box_lower, float), np.asarray(box_upper, float)
    eps = tol * np.max(upper - lower)
    facets = np.asarray(facets)
    out = {}
    for axis in axes:
        on_lo = np.flatnonzero(np.abs(points[:, axis] - lower[axis]) <= eps)
        on_hi = np.flatnonzero(np.abs(points[:, axis] - upper[axis]) <= eps)
        shifted = points[on_hi].copy()
        shifted[:, axis] = lower[axis]
        partner = np.full(len(points), -1, dtype=np.int64)
        dist = np.zeros(0)
        if len(on_lo) and len(on_hi):
            dist, idx = cKDTree(shifted).query(points[on_lo], distance_upper_bound=2 * eps + 1e-300)
            found = np.isfinite(dist)
            partner[on_lo[found]] = on_hi[idx[found]]
            dist = dist[found]
        matched_hi = np.zeros(len(points), dtype=bool)
        matched_hi[partner[partner >= 0]] = True
        lo_facets = facets[np.all(np.isin(facets, on_lo), axis=1)]
        hi_facets = facets[np.all(np.isin(facets, on_hi), axis=1)]
        mapped = partner[lo_facets]
        good = np.all(mapped >= 0, axis=1)
        lo_keys = {tuple(r) for r in np.sort(mapped[good], axis=1)}
        hi_keys = {tuple(r) for r in np.sort(hi_facets, axis=1)}
        out[int(axis)] = {
            'nodes_lower': int(len(on_lo)),
            'nodes_upper': int(len(on_hi)),
            'unmatched_nodes': int(np.count_nonzero(partner[on_lo] < 0)
                                   + np.count_nonzero(~matched_hi[on_hi])),
            'max_distance': float(dist.max()) if len(dist) else 0.0,
            'unmatched_facets': int(np.count_nonzero(~good) + len(lo_keys ^ hi_keys)),
        }
    return out


def anchor_displacement(before: np.ndarray, after: np.ndarray, nodes: np.ndarray) -> dict:
    """How far the given nodes (e.g. quadruple points) moved; 0 when anchors are kept fixed."""
    d = np.linalg.norm(np.asarray(after)[nodes] - np.asarray(before)[nodes], axis=1)
    return {'n': int(len(d)), 'max': float(d.max()) if len(d) else 0.0,
            'mean': float(d.mean()) if len(d) else 0.0, 'moved': int(np.count_nonzero(d > 0))}


# --------------------------------------------------------------------------------------------- #

def _quantiles(values: np.ndarray, upper: bool = False) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {}
    if upper:
        return {'max': float(values.max()), 'p99': float(np.percentile(values, 99)),
                'median': float(np.median(values))}
    return {'min': float(values.min()), 'p1': float(np.percentile(values, 1)),
            'median': float(np.median(values))}


def _volume_deviation(ids: np.ndarray, actual: np.ndarray, reference: Dict[int, float]) -> dict:
    ref = np.array([reference.get(int(g), np.nan) for g in ids])
    keep = np.isfinite(ref) & (ref > 0)
    rel = (actual[keep] - ref[keep]) / ref[keep]
    if not len(rel):
        return {}
    return {'max_abs': float(np.abs(rel).max()), 'p90_abs': float(np.percentile(np.abs(rel), 90)),
            'median': float(np.median(rel)), 'total': float(actual[keep].sum() / ref[keep].sum() - 1)}
