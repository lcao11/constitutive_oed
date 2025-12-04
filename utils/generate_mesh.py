import os, math, tempfile
import numpy as np
import gmsh
import meshio
import dolfin as dl

def build_gmsh_ellipse_rect(rect_width, rect_height, a, b, rotation):
    gmsh.model.add("ellipse_rect")
    occ = gmsh.model.occ
    rect = occ.addRectangle(-rect_width/2, -rect_height/2, 0.0, rect_width, rect_height)
    ell  = occ.addDisk(0.0, 0.0, 0.0, a, b)
    if abs(rotation) > 1e-14:
        occ.rotate([(2, ell)], 0,0,0, 0,0,1, rotation)
    occ.synchronize()
    cut, _ = occ.cut([(2, rect)], [(2, ell)], removeObject=True, removeTool=True)
    occ.synchronize()
    if not cut:
        raise RuntimeError("Boolean cut failed")
    return cut[0][1]

def generate_mesh(comm_mesh,
    rect_width: float,
    rect_height: float,
    stretch: tuple[float, float],
    rotation: float,
    density: int = 40,
    refine_factor: float = 1.0,
    refine_distance: float | None = None,
    corridor_half_width: float | None = None,
    corridor_refine_factor: float = 1.0
) -> dl.Mesh:
    """
    Generate a 2D mesh of a rectangle with an elliptical hole.

    Parameters
    ----------
    refine_factor : refinement near ellipse boundary (Distance/Threshold field).
    corridor_refine_factor : (independent) refinement inside a vertical corridor
        spanning full rectangle height; uses Box field. Use value > 1 to enable.
    corridor_half_width : half width of corridor along x (auto if None).
    """
    a, b = map(float, stretch)
    if a < b:
        a, b = b, a
        rotation += math.pi/2.0
    if refine_distance is None:
        refine_distance = 1.3*max(a,b)

    c = math.cos(rotation)
    s = math.sin(rotation)
    half_w = math.sqrt((a*c)**2 + (b*s)**2)
    half_h = math.sqrt((a*s)**2 + (b*c)**2)
    if corridor_half_width is None:
        corridor_half_width = 1.3 * half_w

    gmsh.initialize([])
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    try:
        surf_tag = build_gmsh_ellipse_rect(rect_width, rect_height, a, b, rotation)

        loop_curves = []
        for dim, cid in gmsh.model.getBoundary([(2, surf_tag)], oriented=False):
            if dim != 1:
                continue
            bb = gmsh.model.getBoundingBox(1, cid)
            cx = 0.5*(bb[0]+bb[3]); cy = 0.5*(bb[1]+bb[4])
            if abs(cx) < 0.75*a and abs(cy) < 0.75*b:
                loop_curves.append(cid)

        h = max(rect_width, rect_height)/float(density)
        overall_refine_cap = max(refine_factor, corridor_refine_factor, 1.0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h / overall_refine_cap)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

        fields = []

        # Ellipse boundary refinement
        if refine_factor > 1.0 and loop_curves:
            d = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(d, "CurvesList", loop_curves)
            gmsh.model.mesh.field.setNumber(d, "NumPointsPerCurve", 40)
            t = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(t, "InField", d)
            gmsh.model.mesh.field.setNumber(t, "SizeMin", h/refine_factor)
            gmsh.model.mesh.field.setNumber(t, "SizeMax", h)
            gmsh.model.mesh.field.setNumber(t, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(t, "DistMax", refine_distance)
            fields.append(t)

        # Independent vertical corridor refinement
        if corridor_refine_factor > 1.0:
            bxf = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(bxf, "VIn", h/corridor_refine_factor)
            gmsh.model.mesh.field.setNumber(bxf, "VOut", h)
            gmsh.model.mesh.field.setNumber(bxf, "XMin", -corridor_half_width)
            gmsh.model.mesh.field.setNumber(bxf, "XMax",  corridor_half_width)
            gmsh.model.mesh.field.setNumber(bxf, "YMin", -rect_height/2)
            gmsh.model.mesh.field.setNumber(bxf, "YMax",  rect_height/2)
            gmsh.model.mesh.field.setNumber(bxf, "ZMin", -1e-3)
            gmsh.model.mesh.field.setNumber(bxf, "ZMax",  1e-3)
            fields.append(bxf)

        if len(fields) == 1:
            gmsh.model.mesh.field.setAsBackgroundMesh(fields[0])
        elif len(fields) > 1:
            mn = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(mn, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(mn)

        gmsh.model.mesh.generate(2)

        with tempfile.TemporaryDirectory() as td:
            msh_path = os.path.join(td, "mesh.msh")
            gmsh.write(msh_path)
            mesh = _load_msh_to_dolfin(comm_mesh, msh_path)
        return mesh
    finally:
        gmsh.finalize()

def _validate_triangles(pts: np.ndarray, tri: np.ndarray, area_tol=1e-14):
    """
    Validate triangle orientation and area.
    Raises RuntimeError if any invalid element found.
    """
    # Compute signed areas: 0.5 * det([[x2-x1, x3-x1],[y2-y1, y3-y1]])
    p1 = pts[tri[:,0]]
    p2 = pts[tri[:,1]]
    p3 = pts[tri[:,2]]
    areas = 0.5 * ((p2[:,0]-p1[:,0])*(p3[:,1]-p1[:,1]) - (p2[:,1]-p1[:,1])*(p3[:,0]-p1[:,0]))
    neg = np.where(areas <= 0)[0]
    tiny = np.where(np.abs(areas) < area_tol)[0]
    if len(neg) > 0:
        raise RuntimeError(f"Found {len(neg)} non-positive area triangles (min area {areas.min():.3e})")
    if len(tiny) > 0:
        raise RuntimeError(f"Found {len(tiny)} near-degenerate triangles (min |area| {np.abs(areas).min():.3e})")

def _load_msh_to_dolfin(comm_mesh, msh_path: str) -> dl.Mesh:
    gm = meshio.read(msh_path)
    tri = None
    for c in gm.cells:
        if c.type in ("triangle", "triangle3") or c.type.startswith("triangle"):
            tri = c.data
            break
    if tri is None:
        raise RuntimeError("No triangle cells")
    if tri.shape[1] > 3:
        raise RuntimeError("High-order triangles detected unexpectedly; regenerate with linear elements")
    pts = gm.points[:, :2]

    # Validate before constructing mesh
    _validate_triangles(pts, tri)

    mesh = dl.Mesh(comm_mesh)
    editor = dl.MeshEditor()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(len(pts))
    editor.init_cells(len(tri))
    for i, (x, y) in enumerate(pts):
        editor.add_vertex(i, [float(x), float(y)])
    for i, (a, b, c) in enumerate(tri):
        editor.add_cell(i, [int(a), int(b), int(c)])
    editor.close()
    return mesh

def check_inclusion(coordinates: np.ndarray, 
                    stretch: np.ndarray, 
                    rotation: float,
                    ) -> np.ndarray:
    """
    Check if the given coordinates are within the ellipse defined by the stretch and rotation.
    Parameters:
    -----------
    coordinates : np.ndarray
        The coordinates to check, shape (N, 2) where N is the number of points.
    stretch : np.ndarray
        Stretch factors for the ellipse [a, b], representing semi-axis lengths.
    rotation : float
        Rotation angle of the ellipse in radians.
    padding : float, optional
        Padding to apply to the ellipse (default: 0.1)

    Returns:
    --------
    np.ndarray
        The indices of the coordinates that are inside the ellipse.
    """
    a, b = stretch[0], stretch[1]
    cos_t = np.cos(rotation)
    sin_t = np.sin(rotation)

    # Transform coordinates to the ellipse's local frame
    x_rotated = (coordinates[:, 0] * cos_t + coordinates[:, 1] * sin_t) / a
    y_rotated = (-coordinates[:, 0] * sin_t + coordinates[:, 1] * cos_t) / b

    # Check if the points are inside the ellipse
    inside_indices = np.where(x_rotated**2 + y_rotated**2 <= 1)[0]

    return inside_indices