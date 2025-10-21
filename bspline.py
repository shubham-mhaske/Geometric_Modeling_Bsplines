import numpy as np

def open_uniform_knot_vector(n, p):
    """
    Generate open uniform knot vector (clamped at endpoints).
    For n+1 control points and degree p, creates knot vector with:
    - First p+1 knots = 0
    - Interior knots uniformly spaced
    - Last p+1 knots = max value
    
    Args:
        n: Number of control points minus 1
        p: Degree of the B-spline
    
    Returns:
        Array of (n+p+2) knot values
    """
    m = n + p + 1
    knots = np.zeros(m + 1, dtype=float)
    knots[p+1:n+1] = np.arange(1, n-p+1)
    knots[n+1:] = n - p + 1
    return knots

def closed_bspline_knot_vector(n, p):
    """
    Generate uniform (periodic) knot vector for closed B-splines.
    
    Args:
        n: Number of control points minus 1 (after wrapping)
        p: Degree of the B-spline
    
    Returns:
        Uniform knot vector
    """
    m = n + p + 1
    return np.arange(0, m + 1, dtype=float)

def find_knot_span(u, knots, p):
    """
    Find the knot span index for parameter u.
    Returns index k such that knots[k] <= u < knots[k+1]
    
    Args:
        u: Parameter value
        knots: Knot vector
        p: Degree
    
    Returns:
        Knot span index k
    """
    n = len(knots) - p - 2
    
    # Handle special cases for u at the ends of the parameter range
    if u >= knots[n+1]:
        return n
    if u <= knots[p]:
        return p
    
    # Binary search for the span
    low, high = p, n + 1
    mid = (low + high) // 2
    
    while u < knots[mid] or u >= knots[mid+1]:
        if u < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    
    return mid

def de_boor(u, p, knots, control_points, show_steps=False):
    """
    de Boor's algorithm for B-spline curve evaluation.
    Implements the recursive blending formula from class notes.
    
    For a B-spline curve of degree p with n+1 control points:
    C(u) = sum_{i=0}^{n} P_i * N_{i,p}(u)
    
    Args:
        u: Parameter value to evaluate
        p: Degree of the B-spline
        knots: Knot vector
        control_points: Array of control points (n+1) x d
        show_steps: If True, return intermediate blending steps
    
    Returns:
        Point on curve C(u), and optionally steps for visualization
    """
    k = find_knot_span(u, knots, p)
    
    # For parameter u in [knots[k], knots[k+1]), we need points P_{k-p} through P_k
    d = [np.array(control_points[j], dtype=float) for j in range(k-p, k+1)]
    steps = [np.array(d)]
    
    # de Boor recursion: r levels of blending
    for r in range(1, p+1):
        for j in range(p, r-1, -1):
            i = k - p + j
            denom = knots[i+p-r+1] - knots[i]
            
            if abs(denom) < 1e-10:
                alpha = 0.0
            else:
                alpha = (u - knots[i]) / denom
            
            d[j] = (1.0 - alpha) * d[j-1] + alpha * d[j]
        
        if show_steps:
            steps.append(np.array(d))
    
    if show_steps:
        return d[p], steps
    return d[p]

def bspline_derivative(u, p, knots, control_points):
    """
    Calculate first derivative of B-spline curve at parameter u.
    Uses the control point differencing formula:
    C'(u) = p * sum_{i=0}^{n-1} (P_{i+1} - P_i)/(t_{i+p+1} - t_{i+1}) * N_{i,p-1}(u)
    
    Args:
        u: Parameter value
        p: Degree
        knots: Knot vector
        control_points: Array of control points
    
    Returns:
        Tangent vector at u
    """
    if p == 0:
        return np.zeros_like(control_points[0])
    
    n = len(control_points) - 1
    
    Q = []
    for i in range(n):
        denom = knots[i+p+1] - knots[i+1]
        if abs(denom) < 1e-10:
            Q.append(np.zeros_like(control_points[0]))
        else:
            Q.append(p * (control_points[i+1] - control_points[i]) / denom)
    
    # Evaluate derivative curve (degree p-1) at u
    return de_boor(u, p-1, knots[1:-1], np.array(Q))

def bspline_second_derivative(u, p, knots, control_points):
    """
    Calculate second derivative of B-spline curve at parameter u.
    
    Args:
        u: Parameter value
        p: Degree
        knots: Knot vector
        control_points: Array of control points
    
    Returns:
        Curvature vector at u
    """
    if p <= 1:
        return np.zeros_like(control_points[0])
    
    n = len(control_points) - 1
    
    # First derivative control points
    Q = []
    for i in range(n):
        denom = knots[i+p+1] - knots[i+1]
        if abs(denom) < 1e-10:
            Q.append(np.zeros_like(control_points[0]))
        else:
            Q.append(p * (control_points[i+1] - control_points[i]) / denom)
    
    # Second derivative control points
    R = []
    for i in range(n-1):
        denom = knots[i+p] - knots[i+1]
        if abs(denom) < 1e-10:
            R.append(np.zeros_like(control_points[0]))
        else:
            R.append((p-1) * (Q[i+1] - Q[i]) / denom)
    
    # Evaluate second derivative curve (degree p-2) at u
    return de_boor(u, p-2, knots[2:-2], np.array(R))

def rational_bspline_point(u, p, knots, control_points, weights):
    """
    Evaluate rational B-spline (NURBS) curve at parameter u.
    Uses homogeneous coordinates: actual point (wx, wy, wz, w) represented as (x, y, z, w).
    
    Args:
        u: Parameter value
        p: Degree
        knots: Knot vector
        control_points: Array of control points
        weights: Array of weights (same length as control_points)
    
    Returns:
        Point on NURBS curve
    """
    # Convert to homogeneous coordinates: [x*w, y*w, z*w, w]
    homo_points = np.array([np.append(pt * w, w) for pt, w in zip(control_points, weights)])
    
    Cw = de_boor(u, p, knots, homo_points)
    
    # Project back to Cartesian: divide by weight
    if abs(Cw[-1]) < 1e-10:
        raise ValueError("Weight denominator is zero in rational evaluation")
    
    return Cw[:-1] / Cw[-1]

def knot_insertion(control_points, knot_vector, degree, u_new):
    """
    Insert a new knot without changing the curve shape (Oslo algorithm #1).
    
    Args:
        control_points: Current control points (n+1 points)
        knot_vector: Current knot vector
        degree: Degree of B-spline
        u_new: Parameter value to insert
    
    Returns:
        Tuple of (new_control_points, new_knot_vector)
    """
    n = len(control_points) - 1
    p = degree
    
    k = find_knot_span(u_new, knot_vector, p)
    multiplicity = np.sum(np.isclose(knot_vector, u_new))
    if multiplicity >= p:
        # Cannot insert knot, multiplicity would be > p
        return control_points, knot_vector

    new_knot_vector = np.insert(knot_vector, k + 1, u_new)

    new_pts = np.zeros((n + 2, control_points.shape[1]))
    
    new_pts[:k-p+1] = control_points[:k-p+1]
    new_pts[k+1:] = control_points[k:]
    
    for i in range(k - p + 1, k + 1):
        denom = knot_vector[i + p] - knot_vector[i]
        alpha = (u_new - knot_vector[i]) / denom if abs(denom) > 1e-10 else 1.0
        new_pts[i] = (1 - alpha) * control_points[i - 1] + alpha * control_points[i]
        
    return new_pts, new_knot_vector

def frenet_frame(u, p, knots, control_points, normalize=True):
    """
    Calculate Frenet frame (tangent, normal, binormal) at parameter u.

    For 2D: Returns (point, tangent, normal, None)
    For 3D: Returns (point, tangent, normal, binormal)

    Args:
        u: Parameter value
        p: Degree
        knots: Knot vector
        control_points: Array of control points
        normalize: If True, returns unit vectors; if False, actual magnitudes (tangent as speed, normal as curvature)

    Returns:
        Tuple: (point, tangent, normal, binormal)
    """
    point = de_boor(u, p, knots, control_points)
    tangent = bspline_derivative(u, p, knots, control_points)
    tangent_mag = np.linalg.norm(tangent)
    if tangent_mag < 1e-10:
        raise ValueError("Tangent vector is zero (singular point)")

    tangent_unit = tangent / tangent_mag

    # --- 2D CASE ---
    if control_points.shape[1] == 2:
        if normalize:
            # Perpendicular unit normal
            normal = np.array([-tangent_unit[1], tangent_unit[0]])
            return point, tangent_unit, normal, None
        else:
            # True principal normal (direction = perpendicular to tangent, magnitude = curvature)
            deriv1 = tangent
            deriv2 = bspline_second_derivative(u, p, knots, control_points)
            # Project 2nd derivative orthogonal to unit tangent
            proj = deriv2 - np.dot(deriv2, tangent_unit) * tangent_unit
            denom = tangent_mag ** 2
            curvature = np.linalg.norm(proj) / denom if denom > 1e-10 else 0.0
            if np.linalg.norm(proj) > 1e-10:
                normal = proj / np.linalg.norm(proj) * curvature
            else:
                normal = np.zeros_like(tangent_unit)
            return point, deriv1, normal, None

    # --- 3D CASE ---
    else:
        deriv1 = tangent
        deriv2 = bspline_second_derivative(u, p, knots, control_points)
        # Binormal: T x C''(t)
        binormal_raw = np.cross(tangent_unit, deriv2)
        binorm_mag = np.linalg.norm(binormal_raw)
        if binorm_mag < 1e-10:
            # If C'' is parallel to T, the binormal is undefined.
            # We create an arbitrary one perpendicular to T.
            if abs(tangent_unit[0]) > 1e-10 or abs(tangent_unit[1]) > 1e-10:
                binormal_unit = np.array([-tangent_unit[1], tangent_unit[0], 0.0])
            else:
                binormal_unit = np.array([1.0, 0.0, 0.0])
            binormal_unit = binormal_unit / np.linalg.norm(binormal_unit)
            binormal = binormal_unit if normalize else np.zeros_like(tangent_unit)
        else:
            binormal_unit = binormal_raw / binorm_mag
            binormal = binormal_unit if normalize else binormal_raw

        # Normal: B x T
        normal_direction = np.cross(binormal_unit, tangent_unit)
        if normalize:
            normal = normal_direction
            return point, tangent_unit, normal, binormal
        else:
            # For unnormalized, magnitude is curvature
            proj = deriv2 - np.dot(deriv2, tangent_unit) * tangent_unit
            denom = tangent_mag ** 2
            curvature = np.linalg.norm(proj) / denom if denom > 1e-10 else 0.0
            normal = normal_direction * curvature
            return point, deriv1, normal, binormal

def closed_bspline(control_points, degree):
    """
    Create closed B-spline by wrapping first 'degree' control points to the end.
    
    Args:
        control_points: Original control points
        degree: Degree of B-spline
    
    Returns:
        Extended control point array for closed curve
    """
    return np.vstack([control_points, control_points[:degree]])