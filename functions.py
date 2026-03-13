import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from meshpy import triangle
# import threding


def make_simple_mesh(a0: float) -> tuple[np.array, np.array]:
    """
    a0: used
        create an numpy: np.array with possions in the xy plane
        give be a array with list 2d connections between the points in the xy plane

    """
    b0 = np.sqrt(3) * a0
    xy = np.array([[0, 0], [b0, 0], [b0 / 2, a0 / 2], [0, a0], [b0, a0]])

    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2],
                     [1, 4], [2, 3], [2, 4], [3, 4]])

    return xy, edges


def make_mesh(Lx, Ly, N):
    points = [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]
    segments = [(0, 1), (1, 2), (2, 3), (3, 0)]

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(segments)

    max_area = (Lx * Ly) / N * np.sqrt(3) / 2
    mesh = triangle.build(mesh_info, max_volume=max_area)

    xy = np.array(mesh.points)

    edges = []
    for tri in mesh.elements:
        tri = sorted(tri)  # Sort the indices to ensure uniqueness
        edges.append((tri[0], tri[1]))
        edges.append((tri[1], tri[2]))
        edges.append((tri[2], tri[0]))
    edges = np.array(list(set(edges)))  # Remove duplicates
    return xy, edges


def plot_mesh(xy, edges):
    fig, ax = plt.subplots(1, 1)

    for edge in edges:
        ax.plot(xy[edge, 0], xy[edge, 1], "k")

    ax.scatter(xy[:, 0], xy[:, 1], s=20, color="red")
    ax.set_aspect("equal")

    plt.show()


def ell0__(nodes, connections): return np.linalg.norm(
    nodes[connections[:, 0]] - nodes[connections[:, 1]], axis=1)


def spring_energy(xy: np.array, edges: np.array, k: float, ell0_: np.array) -> float:
    """
    Calculate the total elastic potential energy of the spring system.

    Parameters
    ----------
    xy : np.array
        Node positions, shape (n_nodes, 2).
    edges : np.array
        Edge connections, shape (n_edges, 2).
    k : float
        Spring constant.
    ell0_ : np.array
        Equilibrium lengths for each edge, shape (n_edges,).

    Returns
    -------
    float
        Total potential energy.
    """
    energy = 0.0
    for edge, ell0 in zip(edges, ell0_):
        i, j = edge
        rij = xy[j] - xy[i]
        ell = np.linalg.norm(rij)
        energy += 0.5 * k * (ell - ell0) ** 2
    return energy


def spring_forces(xy, edges, k, ell0_) -> np.array:
    """
    Calculate the net forces on each node due to the springs.

    Parameters
    ----------
    xy : np.array
        Node positions, shape (n_nodes, 2).
    edges : np.array
        Edge connections, shape (n_edges, 2).
    k : float
        Spring constant.
    ell0_ : np.array
        Equilibrium lengths for each edge, shape (n_edges,).

    Returns
    -------
    np.array
        Forces on each node, shape (n_nodes, 2).
    """
    forces = np.zeros_like(xy)
    for edge, ell0 in zip(edges, ell0_):
        i, j = edge
        rij = xy[j] - xy[i]
        ell = np.linalg.norm(rij)
        f_mag = k * (ell - ell0)
        f_vec = f_mag * (rij / ell)
        forces[i, :] += f_vec
        forces[j, :] -= f_vec
    return forces


def _boundary_ids(nodes_r0):
    """Compute left/right/bottom boundary node indices from node positions."""
    tol = 1e-12
    x_min = nodes_r0[:, 0].min()
    x_max = nodes_r0[:, 0].max()
    y_min = nodes_r0[:, 1].min()
    ids_left = np.where(nodes_r0[:, 0] < x_min + tol)[0]
    ids_right = np.where(nodes_r0[:, 0] > x_max - tol)[0]
    ids_bottom = np.where(nodes_r0[:, 1] < y_min + tol)[0]
    return ids_left, ids_right, ids_bottom


def _boundary_ids_with_top(nodes_r0):
    """Compute left/right/bottom/top boundary node indices from node positions."""
    tol = 1e-12
    x_min = nodes_r0[:, 0].min()
    x_max = nodes_r0[:, 0].max()
    y_min = nodes_r0[:, 1].min()
    y_max = nodes_r0[:, 1].max()
    ids_left = np.where(nodes_r0[:, 0] < x_min + tol)[0]
    ids_right = np.where(nodes_r0[:, 0] > x_max - tol)[0]
    ids_bottom = np.where(nodes_r0[:, 1] < y_min + tol)[0]
    ids_top = np.where(nodes_r0[:, 1] > y_max - tol)[0]
    return ids_left, ids_right, ids_bottom, ids_top


def total_energy_rigid_top(xy_flat, edges, k, K, K_top, ell0_, Lx_plate, nodes_ref,
                           ids_left, ids_right, ids_bottom, ids_top):
    """
    Total energy with an additional rigid-top constraint.

    All top nodes (at y_max) are coupled so they share the same y-coordinate
    — they can collectively move up/down, but not independently. The penalty
    term is K_top * sum_i (y_i - y_bar)^2 where y_bar is the mean y of the
    top nodes, which is zero when all top nodes have the same y.

    Parameters
    ----------
    K_top : float
        Stiffness that enforces the rigid-top coupling.
    ids_top : np.array
        Indices of top-boundary nodes.
    (remaining parameters same as total_energy)
    """
    xy = xy_flat.reshape((-1, 2))
    energy = spring_energy(xy, edges, k, ell0_)
    energy += 0.5 * K * ((xy[ids_left, 0]) ** 2).sum()
    energy += 0.5 * K * ((xy[ids_right, 0] - Lx_plate) ** 2).sum()
    energy += 0.5 * K * \
        ((xy[ids_bottom, 1] - nodes_ref[ids_bottom, 1]) ** 2).sum()
    # Rigid-top: penalise deviation of each top node's y from the group mean
    y_top = xy[ids_top, 1]
    y_bar = y_top.mean()
    energy += K_top * ((y_top - y_bar) ** 2).sum()
    return energy


def total_energy_rigid_top_jacobian(xy_flat, edges, k, K, K_top, ell0_, Lx_plate,
                                    nodes_ref, ids_left, ids_right, ids_bottom, ids_top):
    """
    Jacobian of total_energy_rigid_top.

    Gradient of the rigid-top term: d/dy_i [K_top * sum_j (y_j - y_bar)^2]
    = 2 * K_top * (y_i - y_bar)  for each top node i.
    """
    xy = xy_flat.reshape((-1, 2))
    grad = -spring_forces(xy, edges, k, ell0_)
    grad[ids_left, 0] += K * xy[ids_left, 0]
    grad[ids_right, 0] += K * (xy[ids_right, 0] - Lx_plate)
    grad[ids_bottom, 1] += K * (xy[ids_bottom, 1] - nodes_ref[ids_bottom, 1])
    # Rigid-top gradient
    y_top = xy[ids_top, 1]
    y_bar = y_top.mean()
    grad[ids_top, 1] += 2.0 * K_top * (y_top - y_bar)
    return grad.flatten()


def simulate_rigid_top(stretch_factors=[0.0, 0.0001, 0.01, 0.05, 0.1, 0.3, 0.5],
                       k=1e2,
                       K=1e4,
                       K_top=1e4,
                       task5=False,
                       task6=False,
                       nodes_r0=None,
                       connections=None):
    """
    Like plot_strain_distribution_v2 but with all top nodes rigidly coupled
    in the y-direction: they are free to collectively translate up/down, but
    their y-coordinates are constrained to be equal (rigid top bar).

    Parameters
    ----------
    k : float
        Spring constant.
    K : float
        Penalty stiffness for left/right/bottom boundary conditions.
    K_top : float
        Penalty stiffness for the rigid-top coupling (default: same as K).
    stretch_factors, task5, task6, nodes_r0, connections : same as
        plot_strain_distribution_v2.
    """
    if not task6 and not task5:
        print("No task called")
        return
    if nodes_r0 is None:
        print("No nodes given to the function")
        return
    if connections is None:
        print("No connections given to the function")
        return

    nodes_ref = nodes_r0.copy()

    Lx0 = nodes_ref[:, 0].max() - nodes_ref[:, 0].min()
    Ly0 = nodes_ref[:, 1].max() - nodes_ref[:, 1].min()

    ell0_ = ell0__(nodes_ref, connections)
    ids_left, ids_right, ids_bottom, ids_top = _boundary_ids_with_top(
        nodes_ref)

    print(f"Lx0: {Lx0}, Ly0: {Ly0}  |  top nodes: {len(ids_top)}")

    epsilon_x = epsilon_y = 0.0
    for f in stretch_factors:
        Lx_plate = Lx0 * (1 + f)
        xy_init = nodes_ref.copy()

        res = minimize(
            total_energy_rigid_top,
            xy_init.flatten(),
            args=(connections, k, K, K_top, ell0_, Lx_plate,
                  nodes_ref, ids_left, ids_right, ids_bottom, ids_top),
            method='Newton-CG',
            jac=total_energy_rigid_top_jacobian,
            tol=1e-12,
            options={'maxiter': 1000}
        )
        xy_def = res.x.reshape((-1, 2))

        if not task5:
            strains = []
            for edge, ell0 in zip(connections, ell0_):
                i, j = edge
                rij = xy_def[j] - xy_def[i]
                ell = np.linalg.norm(rij)
                strains.append((ell - ell0) / ell0)
            strains = np.array(strains)

            max_abs = max(np.abs(strains).max(), 1e-10)
            norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
            cmap = cm.coolwarm

            plt.figure(figsize=(6, 5))
            for e_idx, edge in enumerate(connections):
                color = cmap(norm(strains[e_idx]))
                plt.plot(xy_def[edge, 0], xy_def[edge, 1],
                         color=color, linewidth=2)
            plt.scatter(xy_def[:, 0], xy_def[:, 1],
                        s=30, color='black', zorder=5)
            # Highlight the coupled top nodes
            plt.scatter(xy_def[ids_top, 0], xy_def[ids_top, 1],
                        s=60, color='green', zorder=6, label='rigid top')
            plt.legend(fontsize=8)
            plt.gca().set_aspect("equal")
            plt.title(
                f"f = {f:.2f} (L = {Lx0*(1+f):.4f} m) [rigid top]\n"
                f"Blue = compression, Red = tension")
            plt.tight_layout()
            plt.show()

        Lx_def = xy_def[ids_right, 0].mean() - xy_def[ids_left, 0].mean()
        Ly_def = xy_def[:, 1].max() - xy_def[:, 1].min()

        epsilon_x = (Lx_def - Lx0) / Lx0
        epsilon_y = (Ly_def - Ly0) / Ly0

        Fn = K * (Lx_plate - xy_def[ids_right, 0]).sum()

        if task6:
            sigma_n = Fn / (Ly0 * 1.0)
            if abs(epsilon_x) > 1e-14:
                E_measured = sigma_n / epsilon_x
                v = -epsilon_y / epsilon_x
                print(f"f = {f:.4f} | E = {E_measured:.2f} N/m² | v = {v:.4f} "
                      f"(eps_x={epsilon_x:.6f}, eps_y={epsilon_y:.6f})")

    if abs(epsilon_x) > 1e-14:
        return -epsilon_y / epsilon_x
    return None


def total_energy(xy_flat, edges, k, K, ell0_, Lx_plate, nodes_ref, ids_left, ids_right, ids_bottom):
    """
    Calculate the total energy with additional constraints.

    Parameters
    ----------
    xy_flat : np.array
        Flattened node positions, shape (2*n_nodes,).
    edges : np.array
        Edge connections, shape (n_edges, 2).
    k : float
        Spring constant.
    K : float
        Constraint stiffness.
    ell0_ : np.array
        Equilibrium lengths, shape (n_edges,).
    Lx_plate : float
        Plate length in x-direction.
    nodes_ref : np.array
        Reference (undeformed) node positions, shape (n_nodes, 2).
    ids_left : np.array
        Indices of left-boundary nodes.
    ids_right : np.array
        Indices of right-boundary nodes.
    ids_bottom : np.array
        Indices of bottom-boundary nodes.

    Returns
    -------
    float
        Total energy.
    """
    xy = xy_flat.reshape((-1, 2))
    energy = spring_energy(xy, edges, k, ell0_)
    energy += 0.5 * K * ((xy[ids_left, 0]) ** 2).sum()
    energy += 0.5 * K * ((xy[ids_right, 0] - Lx_plate) ** 2).sum()
    energy += 0.5 * K * \
        ((xy[ids_bottom, 1] - nodes_ref[ids_bottom, 1]) ** 2).sum()
    return energy


def total_energy_jacobian(xy_flat, edges, k, K, ell0_, Lx_plate, nodes_ref, ids_left, ids_right, ids_bottom):
    """
    Calculate the Jacobian (gradient) of the total energy.

    Parameters
    ----------
    xy_flat : np.array
        Flattened node positions, shape (2*n_nodes,).
    edges : np.array
        Edge connections, shape (n_edges, 2).
    k : float
        Spring constant.
    K : float
        Constraint stiffness.
    ell0_ : np.array
        Equilibrium lengths, shape (n_edges,).
    Lx_plate : float
        Plate length in x-direction.
    nodes_ref : np.array
        Reference (undeformed) node positions, shape (n_nodes, 2).
    ids_left : np.array
        Indices of left-boundary nodes.
    ids_right : np.array
        Indices of right-boundary nodes.
    ids_bottom : np.array
        Indices of bottom-boundary nodes.

    Returns
    -------
    np.array
        Flattened gradient, shape (2*n_nodes,).
    """
    xy = xy_flat.reshape((-1, 2))
    grad = -spring_forces(xy, edges, k, ell0_)
    grad[ids_left, 0] += K * xy[ids_left, 0]
    grad[ids_right, 0] += K * (xy[ids_right, 0] - Lx_plate)
    grad[ids_bottom, 1] += K * (xy[ids_bottom, 1] - nodes_ref[ids_bottom, 1])
    return grad.flatten()


def plot_strain_distribution(stretch_factors=[0.0, 0.0001, 0.01, 0.05, 0.1, 0.3, 0.5],
                             k=1e2,
                             K=1e4,
                             task5=False,
                             task6=False,
                             nodes_r0=None,
                             connections=None,
                             ):
    """
    Plot the strain distribution in the mesh for various stretch factors.

    This function minimizes the total energy for each stretch factor to find the deformed node positions,
    calculates the strains for each edge, and visualizes the mesh with edges colored according to strain
    (blue for compression, red for tension).

    Parameters
    ----------
    stretch_factors : list of float, optional
        List of stretch factors to apply (default: [0.0, 0.01, 0.05, 0.1, 0.3, 0.5]).
    k : float, optional
        Spring constant (default: 1e2).
    K : float, optional
        Constraint stiffness (default: 1e4).
    task6 : bool, optional
        Compute macroscopic strain ratio instead of plotting (default: False).
    nodes_r0 : np.array, optional
        Reference node positions. Defaults to the simple mesh nodes.
    connections : np.array, optional
        Edge connections. Defaults to the simple mesh connections.

    Returns
    -------
    return v: realtion between the change in x and y axis 
    show
    ----
        Displays plots for each stretch factor.
    """

    if not task6 and not task5:
        print("No task called")
        return
    if nodes_r0 is None:
        print("There is no nodes given to the function, using simple mesh")
        return

    if connections is None:
        print("There is no connections given to the function, using simple mesh")
        return

    nodes_ref = nodes_r0.copy()

    Lx0 = nodes_ref[:, 0].max() - nodes_ref[:, 0].min()
    x0 = nodes_ref[:, 0].max()
    y0 = nodes_ref[:, 1].max()

    ell0_ = ell0__(nodes_ref, connections)
    ids_left, ids_right, ids_bottom = _boundary_ids(nodes_ref)

    print(f"x0: {x0}, y0: {y0}")
    for f in stretch_factors:
        Lx_plate = Lx0 * (1 + f)

        xy_init = nodes_ref.copy()

        res = minimize(
            total_energy,
            xy_init.flatten(),
            args=(connections, k, K, ell0_, Lx_plate,
                  nodes_ref, ids_left, ids_right, ids_bottom),
            method='Newton-CG',
            jac=total_energy_jacobian,
            tol=1e-12,
            options={'maxiter': 1000}
        )
        xy_def = res.x.reshape((-1, 2))

        if not task5:
            strains = []
            for edge, ell0 in zip(connections, ell0_):
                i, j = edge
                rij = xy_def[j] - xy_def[i]
                ell = np.linalg.norm(rij)
                strains.append((ell - ell0) / ell0)
            strains = np.array(strains)

            max_abs = max(np.abs(strains).max(), 1e-10)
            norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
            cmap = cm.coolwarm

            plt.figure(figsize=(6, 5))

            for e_idx, edge in enumerate(connections):
                color = cmap(norm(strains[e_idx]))
                plt.plot(xy_def[edge, 0], xy_def[edge, 1],
                         color=color, linewidth=2)

            plt.scatter(xy_def[:, 0], xy_def[:, 1],
                        s=30, color='black', zorder=5)
            plt.gca().set_aspect("equal")
            plt.title(
                f"f = {f:.2f} (L = {Lx0*(1+f):.4f} m)\nBlue = compression, Red = tension")

            plt.tight_layout()
            plt.show()

        new_max_x, new_max_y = xy_def[:, 0].max(), xy_def[:, 1].max()
        epsilon_x = (new_max_x - x0) / x0
        epsilon_y = (new_max_y - y0) / y0
        Fn = K * (Lx_plate - xy_def[ids_right, 0]).sum()

        if task6:

            Lx = xy_def[ids_right, 0].mean() - xy_def[ids_left, 0].mean()
            Ly = xy_def[:, 1].max() - xy_def[:, 1].min()

            Ly0 = nodes_ref[:, 1].max() - nodes_ref[:, 1].min()

            epsilon_n = (Lx - Lx0) / Lx0
            Fn = K * (Lx_plate - xy_def[ids_right, 0]).sum()
            sigma_n = Fn / (Ly0 * 1.0)   # B = 1 m

            if abs(epsilon_n) > 1e-14:
                E_measured = sigma_n / epsilon_n
                print(f"f = {f:.4f} | E = {
                      E_measured:.2f} N/m² | ν = {-epsilon_y/epsilon_x:.4f}"
                      )

    return epsilon_y / epsilon_x


def plot_strain_distribution_v2(stretch_factors=[0.0, 0.0001, 0.01, 0.05, 0.1, 0.3, 0.5],
                                k=1e2,
                                K=1e4,
                                task5=False,
                                task6=False,
                                nodes_r0=None,
                                connections=None,
                                ):
    """
    Improved version of plot_strain_distribution.

    Fixes the Poisson's ratio (v) calculation by computing epsilon_x and
    epsilon_y from the full dimension change (width/height) rather than from
    the absolute max coordinate divided by x0/y0.

    epsilon_x = (Lx_deformed - Lx0) / Lx0
    epsilon_y = (Ly_deformed - Ly0) / Ly0
    v = -epsilon_y / epsilon_x
    """

    if not task6 and not task5:
        print("No task called")
        return
    if nodes_r0 is None:
        print("There is no nodes given to the function, using simple mesh")
        return
    if connections is None:
        print("There is no connections given to the function, using simple mesh")
        return

    nodes_ref = nodes_r0.copy()

    Lx0 = nodes_ref[:, 0].max() - nodes_ref[:, 0].min()
    Ly0 = nodes_ref[:, 1].max() - nodes_ref[:, 1].min()

    ell0_ = ell0__(nodes_ref, connections)
    ids_left, ids_right, ids_bottom = _boundary_ids(nodes_ref)

    print(f"Lx0: {Lx0}, Ly0: {Ly0}")
    for f in stretch_factors:
        Lx_plate = Lx0 * (1 + f)

        xy_init = nodes_ref.copy()

        res = minimize(
            total_energy,
            xy_init.flatten(),
            args=(connections, k, K, ell0_, Lx_plate,
                  nodes_ref, ids_left, ids_right, ids_bottom),
            method='Newton-CG',
            jac=total_energy_jacobian,
            tol=1e-12,
            options={'maxiter': 10000}
        )
        xy_def = res.x.reshape((-1, 2))

        if not task5:
            strains = []
            for edge, ell0 in zip(connections, ell0_):
                i, j = edge
                rij = xy_def[j] - xy_def[i]
                ell = np.linalg.norm(rij)
                strains.append((ell - ell0) / ell0)
            strains = np.array(strains)

            max_abs = max(np.abs(strains).max(), 1e-10)
            norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
            cmap = cm.coolwarm

            plt.figure(figsize=(6, 5))
            for e_idx, edge in enumerate(connections):
                color = cmap(norm(strains[e_idx]))
                plt.plot(xy_def[edge, 0], xy_def[edge, 1],
                         color=color, linewidth=2)
            plt.scatter(xy_def[:, 0], xy_def[:, 1],
                        s=30, color='black', zorder=5)
            plt.gca().set_aspect("equal")
            plt.title(
                f"f = {f:.2f} (L = {Lx0*(1+f):.4f} m)\nBlue = compression, Red = tension")
            plt.tight_layout()
            plt.show()

        # Use full dimension change for correct strain calculation
        Lx_def = xy_def[ids_right, 0].mean() - xy_def[ids_left, 0].mean()
        Ly_def = xy_def[:, 1].max() - xy_def[:, 1].min()

        epsilon_x = (Lx_def - Lx0) / Lx0
        epsilon_y = (Ly_def - Ly0) / Ly0

        Fn = K * (Lx_plate - xy_def[ids_right, 0]).sum()

        if task6:
            sigma_n = Fn / (Ly0 * 1.0)   # B = 1 m

            if abs(epsilon_x) > 1e-14:
                E_measured = sigma_n / epsilon_x
                v = -epsilon_y / epsilon_x
                print(f"f = {f:.4f} | E = {E_measured:.2f} N/m² | v = {v:.4f} "
                      f"(eps_x={epsilon_x:.6f}, eps_y={epsilon_y:.6f})")

    if abs(epsilon_x) > 1e-14:
        return -epsilon_y / epsilon_x
    return None


def main():
    nodes, connections = make_mesh(0.2, 0.1, 100)
    epsilon = plot_strain_distribution_v2(
        nodes_r0=nodes, connections=connections, task6=True)

    print(
        f"Strain ratio (epsilon_y / epsilon_x) for each stretch factor: {epsilon}")


if __name__ == "__main__":
    main()
