import numpy as np


# ============================================================================
# KNOT VECTOR GENERATION
# ============================================================================

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


# ============================================================================
# DE BOOR ALGORITHM (Core B-spline Evaluation)
# ============================================================================

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
    
    # Special cases
    if u >= knots[n+1]:
        return n
    if u <= knots[p]:
        return p
    
    # Binary search
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
    # Find the knot span
    k = find_knot_span(u, knots, p)
    
    # Initialize with relevant control points
    # For parameter u in [knots[k], knots[k+1]), we need points P_{k-p} through P_k
    d = [np.array(control_points[j], dtype=float) for j in range(k-p, k+1)]
    steps = [np.array(d)]
    
    # de Boor recursion: r levels of blending
    for r in range(1, p+1):
        for j in range(p, r-1, -1):
            # Compute alpha for blending
            i = k - p + j
            denom = knots[i+p-r+1] - knots[i]
            
            if abs(denom) < 1e-10:
                alpha = 0.0
            else:
                alpha = (u - knots[i]) / denom
            
            # Blend: d[j] = (1-alpha)*d[j-1] + alpha*d[j]
            d[j] = (1.0 - alpha) * d[j-1] + alpha * d[j]
        
        steps.append(np.array(d))
    
    if show_steps:
        return d[p], steps
    return d[p]


# ============================================================================
# DERIVATIVES
# ============================================================================

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
    
    # Compute derivative control points
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


# ============================================================================
# RATIONAL B-SPLINES (NURBS)
# ============================================================================

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
    
    # Evaluate in homogeneous space
    Cw = de_boor(u, p, knots, homo_points)
    
    # Project back to Cartesian: divide by weight
    if abs(Cw[-1]) < 1e-10:
        raise ValueError("Weight denominator is zero in rational evaluation")
    
    return Cw[:-1] / Cw[-1]


# ============================================================================
# KNOT INSERTION
# ============================================================================

def knot_insertion(control_points, knot_vector, degree, u_new):
    """
    Insert a new knot without changing the curve shape (Oslo algorithm).
    
    Args:
        control_points: Current control points
        knot_vector: Current knot vector
        degree: Degree of B-spline
        u_new: Parameter value to insert
    
    Returns:
        Tuple of (new_control_points, new_knot_vector)
    """
    n = len(control_points) - 1
    p = degree
    k = find_knot_span(u_new, knot_vector, p)
    
    # Check if knot already has maximum multiplicity
    multiplicity = np.sum(np.isclose(knot_vector, u_new))
    if multiplicity >= p:
        return control_points, knot_vector
    
    # Insert the knot
    new_knot_vector = np.insert(knot_vector, k+1, u_new)
    
    # Compute new control points
    new_control_points = []
    
    # Points before affected region (unchanged)
    for i in range(k - p + 1):
        new_control_points.append(control_points[i].copy())
    
    # Affected region (blend)
    for i in range(k - p + 1, k + 1):
        denom = knot_vector[i + p] - knot_vector[i]
        if abs(denom) < 1e-10:
            alpha = 0.0
        else:
            alpha = (u_new - knot_vector[i]) / denom
        
        new_pt = (1 - alpha) * control_points[i-1] + alpha * control_points[i]
        new_control_points.append(new_pt)
    
    # Points after affected region (unchanged)
    for i in range(k + 1, n + 1):
        new_control_points.append(control_points[i].copy())
    
    return np.array(new_control_points), new_knot_vector


# ============================================================================
# FRENET FRAME
# ============================================================================

def frenet_frame(u, p, knots, control_points, normalize=False):
    """
    Calculate Frenet frame (tangent, normal, binormal) at parameter u.
    
    For 2D: Returns (point, tangent, normal, None)
    For 3D: Returns (point, tangent, normal, binormal)
    
    Args:
        u: Parameter value
        p: Degree
        knots: Knot vector
        control_points: Array of control points
        normalize: If True, return unit vectors; if False, return actual magnitudes
    
    Returns:
        Tuple of (point, tangent, normal, binormal)
    """
    point = de_boor(u, p, knots, control_points)
    tangent = bspline_derivative(u, p, knots, control_points)
    
    # Store original magnitude
    tangent_mag = np.linalg.norm(tangent)
    if tangent_mag < 1e-10:
        raise ValueError("Tangent vector is zero (singular point)")
    
    tangent_unit = tangent / tangent_mag
    
    if control_points.shape[1] == 2:
        # 2D: Normal is perpendicular to tangent
        if normalize:
            normal = np.array([-tangent_unit[1], tangent_unit[0]])
            return point, tangent_unit, normal, None
        else:
            # Keep magnitude proportional to speed
            normal = np.array([-tangent[1], tangent[0]]) / tangent_mag
            return point, tangent, normal, None
    
    else:
        # 3D: Use second derivative for binormal
        second_deriv = bspline_second_derivative(u, p, knots, control_points)
        
        # Binormal = T × C''
        binormal = np.cross(tangent_unit, second_deriv)
        binormal_mag = np.linalg.norm(binormal)
        
        if binormal_mag < 1e-10:
            # Straight line section
            if abs(tangent_unit[0]) > 0.1 or abs(tangent_unit[1]) > 0.1:
                binormal = np.array([-tangent_unit[1], tangent_unit[0], 0.0])
            else:
                binormal = np.array([1.0, 0.0, 0.0])
            binormal = binormal / np.linalg.norm(binormal)
            binormal_mag = 1.0
        else:
            binormal_unit = binormal / binormal_mag
            binormal = binormal_unit if normalize else binormal
        
        # Normal = B × T
        if normalize:
            binormal_for_cross = binormal / np.linalg.norm(binormal)
            normal = np.cross(binormal_for_cross, tangent_unit)
        else:
            normal = np.cross(binormal / binormal_mag, tangent_unit) * binormal_mag
        
        return point, tangent_unit if normalize else tangent, normal, binormal



# ============================================================================
# CLOSED B-SPLINES
# ============================================================================

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
