"""Conservative XY planning bounds of a static USD triangle mesh in a Z band.

The original collider is never changed. Each clipped face is bounded in XY,
then connected components of overlapping face bounds are merged. A component
box may include empty space, but it cannot remove the bounded faces. Separate
low columns need not be joined by a beam entirely above the height band.

``None`` means unsupported/budget exceeded: the caller must preserve its full
collider bound, never treat this as an empty obstacle. ``[]`` means a supported
mesh has no face intersection with the band. Malformed geometry and collision
approximations whose cooked bounds are unknown raise ``MeshSliceError``; the
caller must reject extraction or obtain an actual cooked-collider bound.
This snapshot is not a swept whole-robot collision or safety certificate.
"""
from __future__ import annotations

import math


MAX_FACES = 100_000
MAX_FACE_VERTICES = 32
MAX_INDICES = 1_000_000
MAX_POINTS = 1_000_000
MAX_OVERLAP_COMPARISONS = 2_000_000
NUMERICAL_PADDING_M = 1e-7
DEGENERATE_AXIS_EXPANSION_M = .002


class MeshSliceError(ValueError):
    """Extraction cannot safely establish the planning bound."""


def _point_on_plane(a, b, height):
    delta = tuple(b[j]-a[j] for j in range(3))
    numerator = height-a[2]
    if not all(math.isfinite(x) for x in (*delta, numerator)) or delta[2] == 0:
        raise MeshSliceError("Mesh clipping arithmetic overflow/degeneracy")
    fraction = numerator/delta[2]
    point = tuple(a[j]+fraction*delta[j] for j in range(3))
    if not math.isfinite(fraction) or not all(math.isfinite(x) for x in point):
        raise MeshSliceError("Mesh clipping produced a non-finite intersection")
    return point


def _clip_plane(vertices, height, keep_above):
    """Sutherland-Hodgman clipping of a convex triangle-derived polygon."""
    if not vertices:
        return []
    output = []
    previous = vertices[-1]
    previous_inside = previous[2] >= height if keep_above else previous[2] <= height
    for current in vertices:
        inside = current[2] >= height if keep_above else current[2] <= height
        if inside != previous_inside:
            output.append(_point_on_plane(previous, current, height))
        if inside:
            output.append(current)
        previous, previous_inside = current, inside
    return output


def _slab_points(vertices, low, high):
    if len(vertices) == 3:
        return _clip_plane(_clip_plane(vertices, low, True), high, False)
    # A nonplanar/concave USD n-gon need not use our preferred triangulation.
    # Every triangulation lies in its vertices' convex hull. Vertices in the
    # band plus intersections of ALL vertex pairs with the two planes include
    # every possible hull-edge extremum of that hull's slab intersection.
    points = [p for p in vertices if low <= p[2] <= high]
    for i, a in enumerate(vertices):
        for b in vertices[i+1:]:
            if a[2] == b[2]:
                continue
            for plane in (low, high):
                if min(a[2], b[2]) <= plane <= max(a[2], b[2]):
                    points.append(_point_on_plane(a, b, plane))
    return points


def _face_bounds(points):
    low = [min(p[j] for p in points) for j in range(3)]
    high = [max(p[j] for p in points) for j in range(3)]
    for axis in (0, 1):
        padding = (DEGENERATE_AXIS_EXPANSION_M if high[axis]-low[axis] < 1e-6
                   else NUMERICAL_PADDING_M)
        low[axis] -= padding
        high[axis] += padding
    return low, high


def _merge_components(rectangles):
    """Merge the original rectangle-overlap graph, with a bounded sweep."""
    parents = list(range(len(rectangles)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    active, comparisons = [], 0
    for i in sorted(range(len(rectangles)), key=lambda i: rectangles[i][0][0]):
        low, high, *_ = rectangles[i]
        active = [j for j in active if rectangles[j][1][0] >= low[0]]
        for j in active:
            comparisons += 1
            if comparisons > MAX_OVERLAP_COMPARISONS:
                return None
            other_low, other_high, *_ = rectangles[j]
            if low[1] <= other_high[1] and other_low[1] <= high[1]:
                a, b = find(i), find(j)
                if a != b:
                    parents[b] = a
        active.append(i)
    components = {}
    for i, (low, high, face_index, ngon) in enumerate(rectangles):
        component = components.setdefault(find(i), {"low": low.copy(), "high": high.copy(),
                                                     "faces": [], "ngons": 0})
        component["low"] = [min(a,b) for a,b in zip(component["low"],low)]
        component["high"] = [max(a,b) for a,b in zip(component["high"],high)]
        component["faces"].append(face_index)
        component["ngons"] += ngon
    return sorted(components.values(), key=lambda c: (*c["low"][:2], *c["high"][:2]))


def _rectangles_from_faces(points, counts, indices, zlow, zhigh, *, path):
    """Pure CPU geometry core; world points must be expressed in metres."""
    if not math.isfinite(zlow) or not math.isfinite(zhigh) or zlow >= zhigh:
        raise MeshSliceError("A finite, increasing height band is required")
    if len(counts) > MAX_FACES or len(indices) > MAX_INDICES or len(points) > MAX_POINTS:
        return None
    if not points or not counts or not indices:
        raise MeshSliceError("An empty/missing mesh is not an established empty slice")
    if any(len(p) != 3 or not all(math.isfinite(float(x)) for x in p) for p in points):
        raise MeshSliceError("Mesh world points must be finite XYZ values")
    if any(isinstance(n,bool) or int(n) != n or n < 3 for n in counts):
        raise MeshSliceError("Malformed mesh face vertex count")
    if max(counts) > MAX_FACE_VERTICES:
        return None
    if sum(counts) != len(indices):
        raise MeshSliceError("Face topology counts do not match the index buffer")
    if any(isinstance(i,bool) or int(i) != i or not 0 <= i < len(points) for i in indices):
        raise MeshSliceError("Mesh topology contains an invalid vertex index")
    rectangles, cursor = [], 0
    for face_index, count in enumerate(counts):
        vertices = [points[i] for i in indices[cursor:cursor+count]]
        cursor += count
        clipped = _slab_points(vertices, zlow-NUMERICAL_PADDING_M, zhigh+NUMERICAL_PADDING_M)
        if clipped:
            low, high = _face_bounds(clipped)
            rectangles.append((low, high, face_index, count != 3))
    components = _merge_components(rectangles)
    if components is None:
        return None
    return [{"path": str(path), "subpart_index": index,
             "xy_bounds_m": [*c["low"][:2], *c["high"][:2]],
             "z_bounds_m": [c["low"][2], c["high"][2]],
             "source": "world_mesh_face_z_slab_conservative_overlap_components",
             "height_range_m": [zlow, zhigh], "face_indices": c["faces"],
             "included_face_count": len(c["faces"]), "original_mesh_face_count": len(counts),
             "conservative_ngon_face_count": c["ngons"],
             "degenerate_axis_expansion_m": DEGENERATE_AXIS_EXPANSION_M,
             "numerical_padding_m": NUMERICAL_PADDING_M,
             "physical_collider_unchanged": True} for index, c in enumerate(components)]


def mesh_slice_rectangles(prim, zlow, zhigh):
    """Return conservative metre-valued rectangles, ``None``, or explicit error.

    Only raw triangle-mesh collision (approximation ``none``) is sliced.
    ``convexHull`` returns None for the caller's full mesh AABB fallback: its
    hull is contained by that bound. Other cooked approximations are rejected
    because their extent need not be contained by the original mesh AABB.
    Path always names the original collider; subpart_index is metadata only.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        return None
    approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
    if approximation == "convexHull":
        return None
    if approximation not in (None, "none"):
        raise MeshSliceError(f"Cannot bound cooked mesh approximation {approximation!r} from raw faces")
    mesh = UsdGeom.Mesh(prim)
    attrs = (mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(), mesh.GetFaceVertexIndicesAttr())
    if any(attr.GetNumTimeSamples() > 0 for attr in attrs):
        return None
    ancestor = prim
    while ancestor and not ancestor.IsPseudoRoot():
        if any(op.GetAttr().GetNumTimeSamples() > 0
               for op in UsdGeom.Xformable(ancestor).GetOrderedXformOps()):
            return None
        ancestor = ancestor.GetParent()
    points, counts, indices = [attr.Get(Usd.TimeCode.Default()) for attr in attrs]
    if points is None or counts is None or indices is None:
        raise MeshSliceError("Mesh points/topology are missing")
    if len(counts) > MAX_FACES or len(indices) > MAX_INDICES or len(points) > MAX_POINTS:
        return None
    metres = float(UsdGeom.GetStageMetersPerUnit(prim.GetStage()))
    if not math.isfinite(metres) or metres <= 0:
        raise MeshSliceError("Invalid USD stage distance unit")
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    world_points = [tuple(float(v)*metres for v in transform.Transform(Gf.Vec3d(*p))) for p in points]
    rectangles = _rectangles_from_faces(world_points, list(counts), list(indices),
                                        float(zlow), float(zhigh), path=prim.GetPath())
    if rectangles is not None:
        for rectangle in rectangles:
            rectangle["collision_approximation"] = "none"
    return rectangles
