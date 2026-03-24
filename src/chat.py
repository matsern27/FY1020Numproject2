"""
Spring-mesh simulation for computing elastic deformation and material properties.

Generates a triangular mesh, models edges as linear springs, and computes
equilibrium configurations under uniaxial stretch. Extracts Young's modulus
and Poisson's ratio from the deformed geometry.
"""

from __future__ import annotations

import dataclasses
from enum import Enum, auto

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from meshpy import triangle
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TOLERANCE = 1e-12
BOUNDARY_TOL = 1e-12
UNIT_THICKNESS = 1.0  # out-of-plane thickness B [m]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class SimMode(Enum):
    """What the simulation should produce."""
    PLOT = auto()         # strain colour maps
    MEASURE = auto()      # print E and ν, no plots
    PLOT_AND_MEASURE = auto()


@dataclasses.dataclass
class MeshProblem:
    """All immutable data needed to define and solve the spring problem."""
    nodes_ref: np.ndarray          # (n_nodes, 2) reference positions
    edges: np.ndarray              # (n_edges, 2) connectivity
    rest_lengths: np.ndarray       # (n_edges,)   natural spring lengths
    k: float                       # spring stiffness
    K: float                       # penalty stiffness for boundary constraints
    ids_left: np.ndarray
    ids_right: np.ndarray
    ids_bottom: np.ndarray
    Lx0: float                     # reference width
    Ly0: float                     # reference height

    @classmethod
    def from_mesh(
        cls,
        nodes: np.ndarray,
        edges: np.ndarray,
        k: float = 1e2,
        K: float = 1e4,
    ) -> MeshProblem:
        nodes_ref = nodes.copy()
        rest_lengths = compute_rest_lengths(nodes_ref, edges)
        ids_left, ids_right, ids_bottom = _boundary_ids(nodes_ref)
        Lx0 = nodes_ref[:, 0].max() - nodes_ref[:, 0].min()
        Ly0 = nodes_ref[:, 1].max() - nodes_ref[:, 1].min()
        return cls(
            nodes_ref=nodes_ref,
            edges=edges,
            rest_lengths=rest_lengths,
            k=k,
            K=K,
            ids_left=ids_left,
            ids_right=ids_right,
            ids_bottom=ids_bottom,
            Lx0=Lx0,
            Ly0=Ly0,
        )


@dataclasses.dataclass
class RigidTopProblem(MeshProblem):
    """Extends MeshProblem with rigid-top coupling."""
    ids_top: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([]))
    K_top: float = 1e4

    @classmethod
    def from_mesh(
        cls,
        nodes: np.ndarray,
        edges: np.ndarray,
        k: float = 1e2,
        K: float = 1e4,
        K_top: float = 1e4,
    ) -> RigidTopProblem:
        nodes_ref = nodes.copy()
        rest_lengths = compute_rest_lengths(nodes_ref, edges)
        ids_left, ids_right, ids_bottom, ids_top = _boundary_ids_with_top(
            nodes_ref)
        Lx0 = nodes_ref[:, 0].max() - nodes_ref[:, 0].min()
        Ly0 = nodes_ref[:, 1].max() - nodes_ref[:, 1].min()
        return cls(
            nodes_ref=nodes_ref,
            edges=edges,
            rest_lengths=rest_lengths,
            k=k,
            K=K,
            ids_left=ids_left,
            ids_right=ids_right,
            ids_bottom=ids_bottom,
            Lx0=Lx0,
            Ly0=Ly0,
            ids_top=ids_top,
            K_top=K_top,
        )


@dataclasses.dataclass
class EquilibriumResult:
    """Output of a single equilibrium solve."""
    xy_deformed: np.ndarray   # (n_nodes, 2)
    stretch_factor: float
    epsilon_x: float
    epsilon_y: float
    force_normal: float       # net reaction force on the right boundary
    converged: bool


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------
def make_simple_mesh(a0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a minimal 5-node, 8-edge triangular mesh.

    Parameters
    ----------
    a0 : float
        Characteristic lattice spacing in the y-direction.

    Returns
    -------
    nodes : np.ndarray, shape (5, 2)
    edges : np.ndarray, shape (8, 2)
    """
    b0 = np.sqrt(3) * a0
    nodes = np.array([
        [0, 0], [b0, 0], [b0 / 2, a0 / 2], [0, a0], [b0, a0],
    ])
    edges = np.array([
        [0, 1], [0, 2], [0, 3], [1, 2],
        [1, 4], [2, 3], [2, 4], [3, 4],
    ])
    return nodes, edges


def make_mesh(Lx: float, Ly: float, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a Delaunay triangulation of a rectangular domain.

    Parameters
    ----------
    Lx, Ly : float
        Domain dimensions.
    n_cells : int
        Approximate number of triangular cells.

    Returns
    -------
    nodes : np.ndarray, shape (n_nodes, 2)
    edges : np.ndarray, shape (n_edges, 2)
    """
    points = [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]
    segments = [(0, 1), (1, 2), (2, 3), (3, 0)]

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(segments)

    max_area = (Lx * Ly) / n_cells * np.sqrt(3) / 2
    mesh = triangle.build(mesh_info, max_volume=max_area)

    nodes = np.array(mesh.points)

    edge_set: set[tuple[int, int]] = set()
    for tri in mesh.elements:
        a, b, c = sorted(tri)
        edge_set.update(((a, b), (b, c), (a, c)))
    edges = np.array(sorted(edge_set))
    return nodes, edges


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------
def _boundary_ids(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices of nodes on the left, right, and bottom edges."""
    x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
    y_min = nodes[:, 1].min()
    ids_left = np.where(nodes[:, 0] < x_min + BOUNDARY_TOL)[0]
    ids_right = np.where(nodes[:, 0] > x_max - BOUNDARY_TOL)[0]
    ids_bottom = np.where(nodes[:, 1] < y_min + BOUNDARY_TOL)[0]
    return ids_left, ids_right, ids_bottom


def _boundary_ids_with_top(
    nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return indices of nodes on the left, right, bottom, and top edges."""
    ids_left, ids_right, ids_bottom = _boundary_ids(nodes)
    y_max = nodes[:, 1].max()
    ids_top = np.where(nodes[:, 1] > y_max - BOUNDARY_TOL)[0]
    return ids_left, ids_right, ids_bottom, ids_top


# ---------------------------------------------------------------------------
# Vectorised physics
# ---------------------------------------------------------------------------
def compute_rest_lengths(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Compute the Euclidean length of each edge in the reference configuration."""
    dr = nodes[edges[:, 1]] - nodes[edges[:, 0]]
    return np.linalg.norm(dr, axis=1)


def spring_energy(xy: np.ndarray, edges: np.ndarray, k: float, rest_lengths: np.ndarray) -> float:
    """Total elastic energy of a network of linear springs (vectorised)."""
    dr = xy[edges[:, 1]] - xy[edges[:, 0]]
    lengths = np.linalg.norm(dr, axis=1)
    return 0.5 * k * np.sum((lengths - rest_lengths) ** 2)


def spring_forces(xy: np.ndarray, edges: np.ndarray, k: float, rest_lengths: np.ndarray) -> np.ndarray:
    """Net force on every node from all attached springs (vectorised)."""
    dr = xy[edges[:, 1]] - xy[edges[:, 0]]
    lengths = np.linalg.norm(dr, axis=1, keepdims=True)
    force_magnitudes = k * (lengths - rest_lengths[:, np.newaxis])
    force_vectors = force_magnitudes * (dr / lengths)  # (n_edges, 2)

    forces = np.zeros_like(xy)
    np.add.at(forces, edges[:, 0], force_vectors)
    np.add.at(forces, edges[:, 1], -force_vectors)
    return forces


def compute_edge_strains(
    xy: np.ndarray, edges: np.ndarray, rest_lengths: np.ndarray,
) -> np.ndarray:
    """Engineering strain (ΔL / L₀) for every edge."""
    dr = xy[edges[:, 1]] - xy[edges[:, 0]]
    lengths = np.linalg.norm(dr, axis=1)
    return (lengths - rest_lengths) / rest_lengths


# ---------------------------------------------------------------------------
# Energy & gradient — standard boundary conditions
# ---------------------------------------------------------------------------
def total_energy(xy_flat: np.ndarray, prob: MeshProblem, Lx_plate: float) -> float:
    """
    Total potential energy: spring energy + penalty terms on left, right, bottom.

    Left nodes are penalised toward x = 0, right nodes toward x = Lx_plate,
    bottom nodes toward their reference y-positions.
    """
    xy = xy_flat.reshape((-1, 2))
    E = spring_energy(xy, prob.edges, prob.k, prob.rest_lengths)
    E += 0.5 * prob.K * np.sum(xy[prob.ids_left, 0] ** 2)
    E += 0.5 * prob.K * np.sum((xy[prob.ids_right, 0] - Lx_plate) ** 2)
    E += 0.5 * prob.K * np.sum(
        (xy[prob.ids_bottom, 1] - prob.nodes_ref[prob.ids_bottom, 1]) ** 2
    )
    return E


def total_energy_gradient(xy_flat: np.ndarray, prob: MeshProblem, Lx_plate: float) -> np.ndarray:
    """Gradient of `total_energy` with respect to node positions."""
    xy = xy_flat.reshape((-1, 2))
    grad = -spring_forces(xy, prob.edges, prob.k, prob.rest_lengths)
    grad[prob.ids_left, 0] += prob.K * xy[prob.ids_left, 0]
    grad[prob.ids_right, 0] += prob.K * (xy[prob.ids_right, 0] - Lx_plate)
    grad[prob.ids_bottom, 1] += prob.K * (
        xy[prob.ids_bottom, 1] - prob.nodes_ref[prob.ids_bottom, 1]
    )
    return grad.flatten()


# ---------------------------------------------------------------------------
# Energy & gradient — rigid-top variant
# ---------------------------------------------------------------------------
def total_energy_rigid_top(xy_flat: np.ndarray, prob: RigidTopProblem, Lx_plate: float) -> float:
    """Like `total_energy` but with an extra penalty coupling all top nodes to the same y."""
    xy = xy_flat.reshape((-1, 2))
    E = spring_energy(xy, prob.edges, prob.k, prob.rest_lengths)
    E += 0.5 * prob.K * np.sum(xy[prob.ids_left, 0] ** 2)
    E += 0.5 * prob.K * np.sum((xy[prob.ids_right, 0] - Lx_plate) ** 2)
    E += 0.5 * prob.K * np.sum(
        (xy[prob.ids_bottom, 1] - prob.nodes_ref[prob.ids_bottom, 1]) ** 2
    )
    y_top = xy[prob.ids_top, 1]
    E += prob.K_top * np.sum((y_top - y_top.mean()) ** 2)
    return E


def total_energy_rigid_top_gradient(
    xy_flat: np.ndarray, prob: RigidTopProblem, Lx_plate: float,
) -> np.ndarray:
    """Gradient of `total_energy_rigid_top`."""
    xy = xy_flat.reshape((-1, 2))
    grad = -spring_forces(xy, prob.edges, prob.k, prob.rest_lengths)
    grad[prob.ids_left, 0] += prob.K * xy[prob.ids_left, 0]
    grad[prob.ids_right, 0] += prob.K * (xy[prob.ids_right, 0] - Lx_plate)
    grad[prob.ids_bottom, 1] += prob.K * (
        xy[prob.ids_bottom, 1] - prob.nodes_ref[prob.ids_bottom, 1]
    )
    y_top = xy[prob.ids_top, 1]
    grad[prob.ids_top, 1] += 2.0 * prob.K_top * (y_top - y_top.mean())
    return grad.flatten()


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def solve_equilibrium(
    prob: MeshProblem,
    stretch_factor: float,
    max_iter: int = 10_000,
    tol: float = 1e-12,
) -> EquilibriumResult:
    """
    Find the equilibrium configuration for a given uniaxial stretch.

    Parameters
    ----------
    prob : MeshProblem or RigidTopProblem
        Problem definition.
    stretch_factor : float
        Fractional extension applied to the right boundary.
    max_iter : int
        Maximum Newton-CG iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    EquilibriumResult
    """
    Lx_plate = prob.Lx0 * (1 + stretch_factor)

    is_rigid_top = isinstance(prob, RigidTopProblem)
    energy_fn = total_energy_rigid_top if is_rigid_top else total_energy
    grad_fn = total_energy_rigid_top_gradient if is_rigid_top else total_energy_gradient

    result = minimize(
        energy_fn,
        prob.nodes_ref.flatten(),
        args=(prob, Lx_plate),
        method="Newton-CG",
        jac=grad_fn,
        tol=tol,
        options={"maxiter": max_iter},
    )
    xy_def = result.x.reshape((-1, 2))

    Lx_def = xy_def[prob.ids_right, 0].mean() - xy_def[prob.ids_left, 0].mean()
    Ly_def = xy_def[:, 1].max() - xy_def[:, 1].min()

    epsilon_x = (Lx_def - prob.Lx0) / prob.Lx0
    epsilon_y = (Ly_def - prob.Ly0) / prob.Ly0

    force_normal = prob.K * np.sum(Lx_plate - xy_def[prob.ids_right, 0])

    return EquilibriumResult(
        xy_deformed=xy_def,
        stretch_factor=stretch_factor,
        epsilon_x=epsilon_x,
        epsilon_y=epsilon_y,
        force_normal=force_normal,
        converged=result.success,
    )


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def compute_material_properties(
    eq: EquilibriumResult, Ly0: float,
) -> dict[str, float] | None:
    """
    Derive Young's modulus E and Poisson's ratio ν from an equilibrium result.

    Returns None if the axial strain is too small for a meaningful calculation.
    """
    if abs(eq.epsilon_x) < 1e-14:
        return None
    sigma = eq.force_normal / (Ly0 * UNIT_THICKNESS)
    return {
        "E": sigma / eq.epsilon_x,
        "nu": -eq.epsilon_y / eq.epsilon_x,
        "sigma": sigma,
        "epsilon_x": eq.epsilon_x,
        "epsilon_y": eq.epsilon_y,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_strain(
    prob: MeshProblem,
    eq: EquilibriumResult,
    title_extra: str = "",
    highlight_top: bool = False,
) -> None:
    """
    Visualise the deformed mesh with edges coloured by strain.

    Blue = compression, red = tension.
    """
    strains = compute_edge_strains(
        eq.xy_deformed, prob.edges, prob.rest_lengths)
    max_abs = max(np.abs(strains).max(), 1e-10)
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    cmap = cm.coolwarm

    fig, ax = plt.subplots(figsize=(6, 5))
    for idx, edge in enumerate(prob.edges):
        color = cmap(norm(strains[idx]))
        ax.plot(eq.xy_deformed[edge, 0], eq.xy_deformed[edge, 1],
                color=color, linewidth=2)

    ax.scatter(eq.xy_deformed[:, 0], eq.xy_deformed[:, 1],
               s=30, color="black", zorder=5)

    if highlight_top and isinstance(prob, RigidTopProblem):
        ax.scatter(eq.xy_deformed[prob.ids_top, 0],
                   eq.xy_deformed[prob.ids_top, 1],
                   s=60, color="green", zorder=6, label="rigid top")
        ax.legend(fontsize=8)

    Lx_stretched = prob.Lx0 * (1 + eq.stretch_factor)
    ax.set_aspect("equal")
    ax.set_title(
        f"f = {eq.stretch_factor:.2f}  (L = {Lx_stretched:.4f} m)"
        f"{title_extra}\nBlue = compression, Red = tension"
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------
def run_simulation(
    nodes: np.ndarray,
    edges: np.ndarray,
    stretch_factors: list[float] | None = None,
    mode: SimMode = SimMode.PLOT_AND_MEASURE,
    k: float = 1e2,
    K: float = 1e4,
    rigid_top: bool = False,
    K_top: float = 1e4,
) -> list[EquilibriumResult]:
    """
    Run the stretch simulation over a sequence of stretch factors.

    Parameters
    ----------
    nodes, edges : np.ndarray
        Mesh definition.
    stretch_factors : list[float]
        Fractional extensions to apply.
    mode : SimMode
        Controls whether to plot, print material properties, or both.
    k : float
        Spring stiffness.
    K : float
        Boundary penalty stiffness.
    rigid_top : bool
        If True, couple all top-boundary nodes to share the same y.
    K_top : float
        Penalty stiffness for the rigid-top constraint.

    Returns
    -------
    list[EquilibriumResult]
    """
    if stretch_factors is None:
        stretch_factors = [0.0, 1e-4, 0.0025, 0.005, 0.001, 0.1, 0.5]

    if rigid_top:
        prob = RigidTopProblem.from_mesh(nodes, edges, k=k, K=K, K_top=K_top)
    else:
        prob = MeshProblem.from_mesh(nodes, edges, k=k, K=K)

    should_plot = mode in (SimMode.PLOT, SimMode.PLOT_AND_MEASURE)
    should_measure = mode in (SimMode.MEASURE, SimMode.PLOT_AND_MEASURE)

    results: list[EquilibriumResult] = []
    for factor in stretch_factors:
        eq = solve_equilibrium(prob, factor)
        results.append(eq)

        if should_plot:
            plot_strain(prob, eq, highlight_top=rigid_top)

        if should_measure:
            props = compute_material_properties(eq, prob.Ly0)
            if props is not None:
                print(
                    f"f = {factor:.4f} | E = {props['E']:.2f} N/m² | "
                    f"ν = {props['nu']:.4f}  "
                    f"(ε_x = {props['epsilon_x']:.6f}, ε_y = {
                        props['epsilon_y']:.6f})"
                )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    nodes, edges = make_mesh(0.2, 0.1, 100)

    results = run_simulation(nodes, edges, mode=SimMode.PLOT_AND_MEASURE)

    last = results[-1]
    if abs(last.epsilon_x) > 1e-14:
        nu = -last.epsilon_y / last.epsilon_x
        print(f"\nFinal Poisson's ratio: {nu:.4f}")


if __name__ == "__main__":
    # main()
    print(int(SimMode.PLOT))
