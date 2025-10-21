"""
Interactive B-Spline Curve Visualizer
Streamlit app for visualizing and interacting with B-spline curves.
Supports all assignment features: 2D/3D, rational curves, closed curves,
knot insertion, de Boor visualization, Frenet frames, and animation.
"""
import streamlit as st
import numpy as np
import time
from dataclasses import dataclass

from bspline import (
    open_uniform_knot_vector,
    closed_bspline_knot_vector,
    de_boor,
    rational_bspline_point,
    knot_insertion,
    frenet_frame,
    closed_bspline,
)
from vizualize import (
    plot_bspline,
    plot_spline_segments,
    illustrate_de_boor_steps,
    illustrate_de_boor_steps_separate,
    plot_frenet_frame,
)

st.set_page_config(layout="wide", page_title="B-Spline Visualizer", page_icon="📐")

@dataclass
class AdvancedAnalysisOptions:
    """Dataclass to hold advanced analysis options."""
    show_segments: bool
    show_de_boor: bool
    t_de_boor: float
    show_frenet: bool
    t_frenet: float
    normalize_frenet: bool
    frenet_scale: float
    show_magnitudes: bool

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def set_default_session_state():
    """Set default values in the session state."""
    st.session_state.control_points_2d = np.array([
        [1., 8.], [2., 2.], [5., 1.], [8., 8.], [9., 2.], [11., 1.]
    ])
    st.session_state.control_points_3d = np.array([
        [1., 8., 0.], [2., 2., 5.], [5., 1., 2.],
        [8., 8., -2.], [9., 2., -5.], [11., 1., 0.]
    ])
    st.session_state.knots = open_uniform_knot_vector(5, 3)
    st.session_state.animate = False
    st.session_state.knot_insert_message = None
    st.session_state.knot_type = "Open Uniform"

def initialize_session_state():
    """Initialize the session state if not already set."""
    if 'control_points_2d' not in st.session_state:
        set_default_session_state()

def reset_all():
    """Reset all session state to default values."""
    set_default_session_state()
    st.success("✅ Reset to defaults")
    st.rerun()

# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def display_array(arr, name):
    """Display a formatted numpy array."""
    st.code(f"{name}:\n{np.array2string(arr, precision=3, separator=', ')}")

# ============================================================================
# UI COMPONENT FUNCTIONS
# ============================================================================

def display_rules_and_formulas():
    """Display B-spline mathematical rules and formulas."""
    with st.expander("📚 B-Spline Theory & Formulas"):
        st.markdown("""
            **Notation:**
            - **n+1**: Number of control points
            - **p**: Degree (polynomial degree)
            - **k = p+1**: Order
            - **m+1**: Number of knots
            ---
            **Key Relationship:** `m = n + p + 1`
            **Requirement:** `n+1 > p` (more points than degree)
            **Knot Vector:** Non-decreasing sequence `t_i ≤ t_{i+1}`
        """)

def display_main_controls(degree, curve_dim):
    """Display main controls for curve dimension, degree, and number of points."""
    control_points_key = f"control_points_{curve_dim.lower()}"
    num_points = st.slider(
        "Number of Control Points (n+1)",
        min_value=degree + 1,
        max_value=50,
        value=len(st.session_state[control_points_key])
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎲 Random Points"):
            size = (num_points, 2 if curve_dim == "2D" else 3)
            st.session_state[control_points_key] = np.random.randint(0, 20, size=size).astype(float)
            st.success(f"✅ Generated {num_points} points")
            st.rerun()

    with col2:
        if st.button("🗑️ Clear Points"):
            st.session_state[control_points_key] = np.array([[0., 0.]] if curve_dim == "2D" else [[0., 0., 0.]])
            st.warning("⚠️ Points cleared")
            st.rerun()

def display_point_editor(curve_dim):
    """Display the text area for editing control points."""
    control_points_key = f"control_points_{curve_dim.lower()}"
    control_points = st.session_state[control_points_key]

    if curve_dim == "2D":
        control_str = "\n".join([f"{p[0]},{p[1]}" for p in control_points])
        edited_str = st.text_area("Edit Points (x,y)", control_str, height=150, key="edit_2d")
    else:
        control_str = "\n".join([f"{p[0]},{p[1]},{p[2]}" for p in control_points])
        edited_str = st.text_area("Edit Points (x,y,z)", control_str, height=150, key="edit_3d")

    try:
        points = np.array([list(map(float, line.split(','))) for line in edited_str.split("\n") if line.strip()])
        if points.size == 0:
            points = np.array([[0., 0.]] if curve_dim == "2D" else [[0., 0., 0.]])

        if curve_dim == "2D" and points.shape[1] != 2:
            st.error("❌ 2D points need exactly 2 coordinates")
            st.stop()
        elif curve_dim == "3D" and points.shape[1] != 3:
            st.error("❌ 3D points need exactly 3 coordinates")
            st.stop()

        if not np.array_equal(points, st.session_state[control_points_key]):
            st.session_state[control_points_key] = points
            st.info("ℹ️ Points updated")
    except Exception as e:
        st.error(f"❌ Invalid format: {e}")
        st.stop()

    return st.session_state[control_points_key]

def display_core_requirements():
    """Display core controls for control points and degree."""
    with st.expander("🎯 Control Points & Degree", expanded=True):
        curve_dim = st.radio("Dimension", ("2D", "3D"), index=0, horizontal=True)
        degree = st.slider("Degree (p)", 1, 10, 3)
        display_main_controls(degree, curve_dim)
        control_points = display_point_editor(curve_dim)
        return control_points, curve_dim, degree

def display_curve_properties(n, p):
    """Display knot vector and curve type controls."""
    with st.expander("⚙️ Knot Vector & Curve Type", expanded=True):
        st.subheader("Knot Vector")
        
        knot_type_options = ["Open Uniform", "Uniform", "Custom"]
        knot_type_index = knot_type_options.index(st.session_state.get('knot_type', "Open Uniform"))
        knot_type = st.selectbox("Type", knot_type_options, index=knot_type_index)

        st.info(f"ℹ️ Need **{n + p + 2}** knot values")

        # ✅ Always start from session state if it exists and has correct length
        if len(st.session_state.knots) == n + p + 2:
            default_knots = st.session_state.knots
        else:
            # Generate only if length is wrong
            if knot_type == "Open Uniform":
                default_knots = open_uniform_knot_vector(n, p)
            elif knot_type == "Uniform":
                default_knots = np.arange(n + p + 2, dtype=float)
            else:
                default_knots = open_uniform_knot_vector(n, p)
            st.session_state.knots = default_knots

        knots_str = st.text_input(
            "Knots (comma-separated)", 
            ",".join(map(str, np.round(default_knots, 2))),
            key="knots_input"
        )

        try:
            knots_from_input = np.array([float(k) for k in knots_str.split(",")])
            if len(knots_from_input) != n + p + 2:
                st.error(f"❌ Need {n + p + 2} knots, got {len(knots_from_input)}")
                st.stop()
            if not np.all(np.diff(knots_from_input) >= -1e-10):
                st.error("❌ Knots must be non-decreasing")
                st.stop()
        except ValueError:
            st.error("❌ Invalid knot format")
            st.stop()

        # ✅ ONLY update session state if user manually changed the knots
        if not np.array_equal(knots_from_input, st.session_state.knots):
            st.session_state.knots = knots_from_input
            st.session_state.knot_type = "Custom"

        st.subheader("Curve Type")
        is_rational = st.checkbox("Rational (NURBS)", value=False)
        weights_str = ""
        if is_rational:
            st.info(f"ℹ️ Enter **{n + 1}** weights")
            weights_str = st.text_area("Weights (one per line)", "\n".join(["1.0"] * (n + 1)))

        is_closed = st.checkbox("Closed Curve", value=False)
        if is_closed:
            st.warning(f"⚠️ First {p} points will wrap to end")

        # ✅ Return session state, not local variable!
        return st.session_state.knots, is_rational, weights_str, is_closed


def display_advanced_analysis(t_min, t_max) -> AdvancedAnalysisOptions:
    """Display advanced analysis controls."""
    with st.expander("🔬 Advanced Analysis & Visualization"):
        show_segments = st.checkbox("Show Spline Segments", value=False)
        st.markdown("---")

        show_de_boor = st.checkbox("Show de Boor Steps", value=False)
        t_de_boor = st.number_input("Parameter for de Boor", min_value=float(t_min), max_value=float(t_max), value=float(t_min + t_max) / 2, disabled=not show_de_boor)
        st.markdown("---")

        show_frenet = st.checkbox("Show Frenet Frame", value=False)
        t_frenet = st.number_input("Parameter for Frenet", min_value=float(t_min), max_value=float(t_max), value=float(t_min + t_max) / 2, disabled=not show_frenet)

        if show_frenet:
            st.markdown("**Frenet Frame Options:**")
            normalize_frenet = st.checkbox("Normalize to unit vectors", value=True, help="Normalize vectors to length 1.")
            frenet_scale = st.slider("Arrow scale", 0.5, 5.0, 1.5, 0.1, help="Visual scaling for arrows.")
            show_magnitudes = st.checkbox("Show vector magnitudes", value=True, help="Display magnitudes in legend.")
        else:
            normalize_frenet, frenet_scale, show_magnitudes = True, 1.5, True

        st.markdown("---")
        st.subheader("Animation")
        if st.button("▶️ Animate Frenet Frame"):
            st.session_state.animate = True

        return AdvancedAnalysisOptions(show_segments, show_de_boor, t_de_boor, show_frenet, t_frenet, normalize_frenet, frenet_scale, show_magnitudes)

def display_knot_insertion(t_min, t_max, degree, control_points):
    """Display knot insertion controls."""
    with st.expander("🔧 Knot Insertion"):
        st.info("ℹ️ Insert a knot without changing the curve's shape.")

        if st.session_state.knot_insert_message:
            msg_type, msg_text = st.session_state.knot_insert_message
            getattr(st, msg_type)(msg_text)

        # ✅ Show current state
        st.write(f"**Current:** {len(control_points)} points, {len(st.session_state.knots)} knots")

        u_insert = st.number_input("Knot value to insert", 
                                   min_value=float(t_min), max_value=float(t_max), 
                                   value=float(t_min + t_max) / 2)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Insert Knot", use_container_width=True):
                try:
                    old_len_pts = len(control_points)
                    old_len_knots = len(st.session_state.knots)

                    # before inserting
                    print("KNOT INSERT DEBUG — before insert")
                    print("u_insert:", u_insert)
                    print("current knots:", np.array2string(st.session_state.knots, precision=3))
                    print("current control points shape:", control_points.shape)

                    # call insertion
                    new_pts, new_knots = knot_insertion(control_points, st.session_state.knots, degree, u_insert)

                    # Update session state with new curve data
                    dim_key = f"control_points_{'2d' if control_points.shape[1] == 2 else '3d'}"
                    st.session_state[dim_key] = new_pts
                    st.session_state.knots = new_knots
                    st.session_state.knot_type = "Custom"

                    # Format the debug info for UI display
                    debug_output = f"""KNOT INSERT DEBUG — before insert
u_insert: {u_insert}
current knots: {np.array2string(control_points, precision=3)}
current control points shape: {control_points.shape}

KNOT INSERT DEBUG — after insert
new knots: {np.array2string(new_knots, precision=3)}
new control points shape: {new_pts.shape}
{np.array2string(new_pts, precision=3, separator=', ')}
"""

                    st.session_state.knot_insert_message = ("code", debug_output)
                except Exception as e:
                    st.session_state.knot_insert_message = ("error", f"❌ Failed: {e}\n{str(e.__traceback__)}")
                st.rerun()
        with col2:
            if st.button("Clear Message", use_container_width=True):
                st.session_state.knot_insert_message = None
                st.rerun()

def display_curve_summary(n, p, knots, control_points, is_rational, is_closed):
    """Display a summary of the curve's configuration."""
    st.subheader("📊 Configuration")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Points", n + 1)
    col2.metric("Degree", p)
    col3.metric("Knots", len(knots))
    col4.metric("Dim", f"{control_points.shape[1]}D")

    col_a, col_b = st.columns(2)
    col_a.metric("Rational", "Yes" if is_rational else "No")
    col_b.metric("Closed", "Yes" if is_closed else "No")

    st.info(f"**Parameter range:** t ∈ [{knots[p]:.3f}, {knots[n+1]:.3f}]")

def calculate_curve_points(t_values, degree, knots, control_points, is_rational, weights_str):
    """Calculate the points on the B-spline curve."""
    if is_rational:
        try:
            weights = np.array([float(w) for w in weights_str.split("\n")])
            if len(weights) != len(control_points):
                st.error(f"❌ Need {len(control_points)} weights, got {len(weights)}")
                st.stop()
        except ValueError:
            st.error("❌ Invalid weights format")
            st.stop()
        return np.array([rational_bspline_point(t, degree, knots, control_points, weights) for t in t_values])
    else:
        return np.array([de_boor(t, degree, knots, control_points) for t in t_values])

def main():
    """Main function to run the Streamlit application."""
    st.title("📐 Interactive B-Spline Curve Visualizer")
    if st.button("🔄 Reset", use_container_width=True):
        reset_all()
    st.markdown("---")

    initialize_session_state()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("🎛️ Controls")
        display_rules_and_formulas()
        control_points, curve_dim, degree = display_core_requirements()

        n = len(control_points) - 1
        if n < degree:
            st.warning("⚠️ Need at least degree+1 control points")
            st.stop()

        knots, is_rational, weights_str, is_closed = display_curve_properties(n, degree)

        if is_closed:
            control_points_to_draw = closed_bspline(control_points, degree)
            n_draw = len(control_points_to_draw) - 1
            knots = closed_bspline_knot_vector(n_draw, degree)
            st.success(f"✅ Closed: {n+1} → {n_draw+1} points")
        else:
            control_points_to_draw, n_draw = control_points, n

        t_min, t_max = knots[degree], knots[n_draw + 1]
        analysis_options = display_advanced_analysis(t_min, t_max)

        if not is_closed:
            display_knot_insertion(t_min, t_max, degree, control_points)

    with col2:
        st.header("📈 Visualization")
        
        # ✅ ALWAYS get fresh values from session state for display
        display_control_points = st.session_state[f'control_points_{curve_dim.lower()}']
        display_knots = st.session_state['knots']
        
        # If closed, wrap the points
        if is_closed:
            display_control_points = closed_bspline(display_control_points, degree)
            n_display = len(display_control_points) - 1
            display_knots = closed_bspline_knot_vector(n_display, degree)
        else:
            n_display = len(display_control_points) - 1
        
        display_curve_summary(n_display, degree, display_knots, display_control_points, is_rational, is_closed)
        
        # Show current state
        st.markdown("**Current Control Points:**")
        display_array(display_control_points, "P")
        
        st.markdown("**Current Knots:**")
        display_array(display_knots, "U")

        st.subheader("Main Curve")
        plot_placeholder = st.empty()

        # ✅ Calculate t_values and curve with fresh display values
        t_values = np.linspace(display_knots[degree], display_knots[n_display + 1], 300)
        curve_points = calculate_curve_points(t_values, degree, display_knots, display_control_points, is_rational, weights_str)

        if st.session_state.animate:
            for t_anim in np.linspace(t_min, t_max, 100):
                fig, ax = plot_bspline(display_control_points, curve_points)
                point, tangent, normal, binormal = frenet_frame(t_anim, degree, display_knots, display_control_points, normalize=analysis_options.normalize_frenet)
                plot_frenet_frame(point, tangent, normal, binormal, ax=ax, scale=analysis_options.frenet_scale, show_magnitudes=analysis_options.show_magnitudes)
                plot_placeholder.pyplot(fig)
                time.sleep(0.05)
            st.session_state.animate = False
            st.rerun()
        else:
            fig, ax = plot_bspline(display_control_points, curve_points)
            if analysis_options.show_segments:
                plot_spline_segments(curve_points, display_knots, degree, ax=ax)

            de_boor_steps = None
            if analysis_options.show_de_boor:
                _, de_boor_steps = de_boor(analysis_options.t_de_boor, degree, display_knots, display_control_points, show_steps=True)
                illustrate_de_boor_steps(de_boor_steps, ax=ax)

            if analysis_options.show_frenet:
                point, tangent, normal, binormal = frenet_frame(analysis_options.t_frenet, degree, display_knots, display_control_points, normalize=analysis_options.normalize_frenet)
                plot_frenet_frame(point, tangent, normal, binormal, ax=ax, scale=analysis_options.frenet_scale, show_magnitudes=analysis_options.show_magnitudes)
                with col1:
                    st.subheader("Frenet Vectors")
                    st.write("**T (Tangent):**", np.round(tangent, 3))
                    st.write("**N (Normal):**", np.round(normal, 3))
                    if binormal is not None:
                        st.write("**B (Binormal):**", np.round(binormal, 3))
                    if not analysis_options.normalize_frenet:
                        st.markdown("**Magnitudes:**")
                        st.write(f"- |T| = {np.linalg.norm(tangent):.4f} (speed)")
                        st.write(f"- |N| = {np.linalg.norm(normal):.4f}")
                        if binormal is not None:
                            st.write(f"- |B| = {np.linalg.norm(binormal):.4f}")

            plot_placeholder.pyplot(fig)

            if de_boor_steps:
                st.markdown("---")
                st.subheader("🔍 de Boor Levels (Individual)")
                st.info(f"ℹ️ Showing {len(de_boor_steps)} recursion levels at t={analysis_options.t_de_boor:.3f}")
                fig_levels = illustrate_de_boor_steps_separate(de_boor_steps)
                st.pyplot(fig_levels)
                with st.expander("📊 Level Coordinates"):
                    for r, pts in enumerate(de_boor_steps):
                        st.write(f"**Level {r}:**")
                        st.code(np.array2string(pts, precision=3, separator=', '))

if __name__ == "__main__":
    main()
