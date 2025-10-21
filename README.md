# Interactive B-Spline Curve Visualizer

**Live Demo:** [https://geometricmodelingbsplinesgit-gpt4sse7xnqh6kq5gw4rrb.streamlit.app/](https://geometricmodelingbsplinesgit-gpt4sse7xnqh6kq5gw4rrb.streamlit.app/)

This project is a Streamlit web application for interactively visualizing and manipulating B-spline curves. It serves as a comprehensive tool for understanding the core concepts of B-splines, including their mathematical properties and behavior.

## Features

*   **2D and 3D Visualization:** Create and manipulate B-spline curves in both 2D and 3D space.
*   **Interactive Controls:** Interactively modify control points, degree, and knot vectors.
*   **Curve Types:** Supports open, closed, and rational (NURBS) B-spline curves.
*   **Advanced Analysis:**
    *   **Knot Insertion:** Insert new knots into the knot vector without changing the curve's shape.
    *   **de Boor's Algorithm:** Visualize the intermediate steps of de Boor's algorithm.
    *   **Frenet Frame:** Display and animate the Frenet frame (Tangent, Normal, Binormal) along the curve.

## File Structure

The project is organized into the following files:

*   `main.py`: This is the main entry point of the Streamlit application. It handles the user interface, application state, and coordinates the interaction between the user, the B-spline library, and the visualization module.

*   `bspline.py`: This file is the core mathematical library of the project. It contains all the algorithms and functions related to B-spline curves, including:
    *   Knot vector generation.
    *   de Boor's algorithm for curve evaluation.
    *   Calculation of derivatives.
    *   Rational B-spline (NURBS) evaluation.
    *   Knot insertion (Oslo algorithm).
    *   Frenet frame calculation.

*   `vizualize.py`: This module is responsible for all the plotting and visualization. It uses Matplotlib to generate the plots for the B-spline curve, control polygon, de Boor's algorithm steps, and the Frenet frame.

*   `requirements.txt`: This file lists the Python dependencies required to run the project (`streamlit`, `numpy`, `matplotlib`).

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run main.py
    ```

## Deployment

This application can be easily deployed using [Streamlit Community Cloud](https://share.streamlit.io/). Simply connect your GitHub repository and select `main.py` as the main file.
