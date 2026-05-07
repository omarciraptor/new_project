# ============================================================
# Two-Higgs-Doublet Model in Bilinear Variables K^mu
# Complete SageMath Script
# ============================================================
#
# PURPOSE
# -------
# This script studies the scalar potential of the two-Higgs-doublet
# model (2HDM) using bilinear variables K^mu = (K0, K1, K2, K3).
#
# The main goal is to compute stationary points on the lightlike
# boundary defined by:
#
#     -K0^2 + K1^2 + K2^2 + K3^2 = 0
#
# using the method of Lagrange multipliers and Groebner bases.
#
# ------------------------------------------------------------
# REQUIRED INSTALLATION
# ------------------------------------------------------------
#
# 1. SageMath is required.
#    This script is meant to be executed with SageMath, not with
#    standard Python, because it uses:
#
#       - var(...)
#       - QQ, RR, CC, QQbar
#       - PolynomialRing(...)
#       - ideal(...).groebner_basis()
#       - variety(QQbar)
#       - Sage symbolic expressions and exact arithmetic
#
#    Recommended ways to run it:
#
#       Option A: Local SageMath installation
#           sage two_higgs_doublet_bilinear_complete.sage
#
#       Option B: CoCalc or a SageMath notebook
#           Copy the content into a SageMath worksheet/notebook.
#
#       Option C: Conda environment with SageMath
#           conda create -n sage-env -c conda-forge sage matplotlib
#           conda activate sage-env
#           sage two_higgs_doublet_bilinear_complete.sage
#
# 2. matplotlib is optional.
#    It is only needed if you want to plot scan results.
#
# ------------------------------------------------------------
# REQUIRED MODEL PARAMETERS
# ------------------------------------------------------------
#
# The script requires numerical values for the following parameters:
#
#     l1, l2, l3, l4, l5, l6, l7, v1, v2, xi
#
# Meaning:
#
#     l1, l2, l3, l4, l5, l6, l7  : quartic couplings of the model
#     v1, v2                      : vacuum expectation values
#     xi                          : relative phase angle
#
# IMPORTANT:
# ----------
# Groebner basis computations are much more stable with exact values.
# Therefore, use rational numbers such as:
#
#     QQ(1)/10       instead of 0.1
#     QQ(30)         instead of 30.0
#     QQ(0)          instead of 0.0
#
# Example parameter dictionary:
#
#     params_example = {
#         l1: QQ(0),
#         l2: QQ(0),
#         l3: QQ(1)/10,
#         l4: QQ(2)/10,
#         l5: QQ(4)/10,
#         l6: QQ(4)/10,
#         l7: QQ(0),
#         v1: QQ(30),
#         v2: QQ(171),
#         xi: QQ(0)
#     }
#
# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------
#
# For a fixed parameter point, the script returns:
#
#     - Real lightlike stationary points
#     - The Lagrange multiplier u
#     - The value of the potential V at each stationary point
#     - The Groebner basis used to solve the system
#
# For a parameter scan, the script returns:
#
#     - The list of stationary values of V for each scanned parameter
#     - The minimum value of V at each point
#     - The number of real solutions found at each point
#
# ============================================================

# ------------------------------------------------------------
# 1. Imports
# ------------------------------------------------------------
import itertools


# ------------------------------------------------------------
# 2. Symbolic variables and model parameters
# ------------------------------------------------------------
# Bilinear variables and Lagrange multiplier.
# K0, K1, K2, K3 represent the four bilinear coordinates K^mu.
# u is the Lagrange multiplier used for the light-cone constraint.
var('K0 K1 K2 K3 u')

# Scalar-potential parameters.
# These are the parameters that must be assigned numerical values
# before solving the system.
var('l1 l2 l3 l4 l5 l6 l7')
var('v1 v2 xi')

# Global variable containers.
# They are used throughout the script to keep the order consistent.
K_VARS = (K0, K1, K2, K3)
MODEL_PARAMS = (l1, l2, l3, l4, l5, l6, l7, v1, v2, xi)


# ------------------------------------------------------------
# 3. Information helpers
# ------------------------------------------------------------
def print_installation_notes():
    """
    Print the installation requirements needed to run this script.

    This function is useful when sharing the script with another person,
    because it clearly explains that the code must be executed with
    SageMath and not with standard Python.
    """
    print("Required installation:")
    print("  1. SageMath")
    print("     Run this script with:")
    print("       sage two_higgs_doublet_bilinear_complete.sage")
    print("  2. matplotlib is optional and only needed for plots.")
    print("\nRecommended Conda command:")
    print("  conda create -n sage-env -c conda-forge sage matplotlib")
    print("  conda activate sage-env")
    print("  sage two_higgs_doublet_bilinear_complete.sage")


def print_required_parameters():
    """
    Print the list of model parameters that must be assigned values.
    """
    print("Required model parameters:")
    for p in MODEL_PARAMS:
        print(f"  - {p}")
    print("\nUse exact rational values whenever possible, for example:")
    print("  QQ(1)/10 instead of 0.1")
    print("  QQ(30)   instead of 30.0")


def parameter_template():
    """
    Return a dictionary template with all required parameters.

    The values are set to None so the user can fill them manually.
    This is mainly a convenience function to avoid forgetting parameters.
    """
    return {p: None for p in MODEL_PARAMS}


# ------------------------------------------------------------
# 4. Small utility functions
# ------------------------------------------------------------
def scalar_part(expr):
    """
    Return the scalar entry of a 1x1 matrix.

    In SageMath, products such as:

        row_vector * matrix * column_vector

    often return a 1x1 matrix instead of a scalar expression. For later
    symbolic manipulation, it is better to extract the unique entry.

    If the input is not a 1x1 matrix, the function returns the input
    unchanged.
    """
    try:
        if hasattr(expr, 'nrows') and hasattr(expr, 'ncols'):
            if expr.nrows() == 1 and expr.ncols() == 1:
                return expr[0, 0]
    except Exception:
        pass
    return expr


def is_almost_real(z, tol=1e-10):
    """
    Check whether a SageMath number is real up to numerical tolerance.

    Groebner basis computations over QQbar can return algebraic numbers
    whose numerical approximation has a very small imaginary part. This
    function treats such tiny imaginary parts as numerical noise.
    """
    zc = CC(z)
    return abs(zc.imag()) <= tol


def real_part(z):
    """
    Return the real part of a SageMath number as an RR value.
    """
    return RR(CC(z).real())


def validate_parameter_dictionary(params):
    """
    Validate that all required model parameters have numerical values.

    Parameters
    ----------
    params : dict
        Dictionary whose keys must include all elements of MODEL_PARAMS:
        l1, l2, l3, l4, l5, l6, l7, v1, v2, xi.

    Notes
    -----
    For Groebner basis computations, exact rational values are preferred.
    Python floats can lead to unstable coercions into polynomial rings.
    """
    missing = [p for p in MODEL_PARAMS if p not in params]
    if missing:
        raise ValueError(f"Missing parameter values: {missing}")

    none_values = [p for p in MODEL_PARAMS if params[p] is None]
    if none_values:
        raise ValueError(f"These parameters are still None: {none_values}")

    float_like = [p for p in MODEL_PARAMS if isinstance(params[p], float)]
    if float_like:
        print("Warning: some parameters were given as Python floats.")
        print("For Groebner bases, exact values such as QQ(1)/10 are safer.")
        print("Float-like parameters:", float_like)


# ------------------------------------------------------------
# 5. Build the 2HDM bilinear potential
# ------------------------------------------------------------
def build_2hdm_bilinear_potential():
    """
    Build the two-Higgs-doublet scalar potential in bilinear variables.

    The potential is written as:

        V(K) = xi_0 K0 + xi_vec . K_vec
             + eta_00 K0^2 + 2 eta K0 K3
             + K_vec^T E K_vec

    where:

        K_vec = (K1, K2, K3)^T

    The function returns both the symbolic potential and the light-cone
    constraint.

    Returns
    -------
    V : Sage symbolic expression
        The symbolic scalar potential in K variables.

    lightcone_constraint : Sage symbolic expression
        The lightlike constraint:

            -K0^2 + K1^2 + K2^2 + K3^2 = 0
    """
    # Spatial bilinear vector K_vec = (K1, K2, K3)^T.
    K_col = matrix(SR, 3, 1, [K1, K2, K3])
    K_row = matrix(SR, 1, 3, [K1, K2, K3])

    # Coefficient of K0^2.
    eta_00 = QQ(1)/4*l1 + QQ(1)/4*l2 + l3 + QQ(1)/4*l4

    # Coefficient of the mixed term 2*K0*K3.
    eta = QQ(1)/4*(l1 - l2)

    # Linear coefficient multiplying K0.
    xi_0 = -l1*v1^2 - l2*v2^2 - 2*l3*v1^2 - 2*l3*v2^2

    # Linear coefficients multiplying K1, K2, K3.
    xi_vec = matrix(SR, 3, 1, [
        -cos(xi)*v1*v2*l5 - QQ(1)/2*l7*v1*v2*sin(xi),
        -sin(xi)*v1*v2*l6 - QQ(1)/2*cos(xi)*v1*v2*l7,
        -v1^2*l1 + v2^2*l2
    ])

    # Symmetric matrix defining the quadratic form in K1, K2, K3.
    E = QQ(1)/8 * matrix(SR, [
        [2*(l5 - l4), l7, 0],
        [l7, 2*(l6 - l4), 0],
        [0, 0, 2*(l1 + l2 - l4)]
    ])

    # Linear and quadratic contributions.
    linear_part = xi_0*K0 + scalar_part(K_row * xi_vec)
    quadratic_part = eta_00*K0^2 + 2*eta*K0*K3 + scalar_part(K_row * E * K_col)

    # Final symbolic potential.
    V = expand(linear_part + quadratic_part)

    # Light-cone boundary condition.
    lightcone_constraint = -K0^2 + K1^2 + K2^2 + K3^2

    return V, lightcone_constraint


# ------------------------------------------------------------
# 6. Construct the Lagrange system
# ------------------------------------------------------------
def build_lagrange_system(params):
    """
    Construct the polynomial system for the constrained problem.

    The Lagrange function is:

        L(K,u) = V(K) + u*g(K)

    where:

        g(K) = -K0^2 + K1^2 + K2^2 + K3^2

    The stationarity system is:

        dL/dK0 = 0
        dL/dK1 = 0
        dL/dK2 = 0
        dL/dK3 = 0
        g(K)   = 0

    Parameters
    ----------
    params : dict
        Numerical values for l1, l2, l3, l4, l5, l6, l7, v1, v2, xi.

    Returns
    -------
    equations : list
        Polynomial equations of the Lagrange system.

    V_num : Sage symbolic expression
        Potential after substituting numerical parameters.

    constraint : Sage symbolic expression
        Light-cone constraint.
    """
    validate_parameter_dictionary(params)

    V_symbolic, constraint = build_2hdm_bilinear_potential()

    # Substitute numerical model parameters only once.
    V_num = expand(V_symbolic.subs(params))

    # Lagrange function.
    L = expand(V_num + u*constraint)

    # Derivative equations with respect to K0, K1, K2, K3.
    gradient_equations = [expand(diff(L, varK)) for varK in K_VARS]

    # Full constrained system.
    equations = gradient_equations + [constraint]

    return equations, V_num, constraint


# ------------------------------------------------------------
# 7. Groebner basis and lightlike solution extraction
# ------------------------------------------------------------
def solve_lightlike_extrema(params, field=QQbar, real_only=True, tol=1e-10, verbose=True):
    """
    Solve the lightlike stationary points using Groebner bases.

    Parameters
    ----------
    params : dict
        Dictionary with numerical values for l1, l2, l3, l4, l5, l6,
        l7, v1, v2 and xi.

    field : Sage field
        Coefficient field for the polynomial ring.
        - QQ is appropriate for purely rational systems.
        - QQbar is safer when algebraic numbers appear.

    real_only : bool
        If True, keep only real K-solutions.

    tol : float
        Numerical tolerance used to decide whether a solution is real.

    verbose : bool
        If True, print diagnostic information.

    Returns
    -------
    results : list of dict
        Each dictionary contains:
            K0, K1, K2, K3, u, V

    G : Groebner basis
        Groebner basis of the generated ideal.
    """
    equations, V_num, constraint = build_lagrange_system(params)

    # Polynomial ring with lexicographic order.
    # The variable order is important for elimination.
    R = PolynomialRing(field, names=('K0', 'K1', 'K2', 'K3', 'u'), order='lex')
    K0R, K1R, K2R, K3R, uR = R.gens()

    # Convert the symbolic equations into polynomials in R.
    equations_R = [R(eq) for eq in equations]

    # Ideal and Groebner basis.
    I = R.ideal(equations_R)
    G = I.groebner_basis()

    if verbose:
        print("Groebner basis computed.")
        print("Number of basis elements:", len(G))
        print("Ideal dimension:", I.dimension())

    # For a zero-dimensional ideal, variety(QQbar) returns all algebraic solutions.
    if I.dimension() != 0:
        print("Warning: the ideal is not zero-dimensional.")
        print("The system may have infinitely many solutions or special parameter degeneracies.")
        return [], G

    raw_solutions = I.variety(QQbar)

    results = []
    for sol in raw_solutions:
        K_values = [sol[K0R], sol[K1R], sol[K2R], sol[K3R]]
        u_value = sol[uR]

        # Keep only real solutions if requested.
        if real_only:
            if not all(is_almost_real(z, tol) for z in K_values + [u_value]):
                continue
            K_values = [real_part(z) for z in K_values]
            u_value = real_part(u_value)

        substitution_K = {
            K0: K_values[0],
            K1: K_values[1],
            K2: K_values[2],
            K3: K_values[3]
        }

        # Evaluate the numerical potential at the stationary point.
        V_value = real_part(V_num.subs(substitution_K)) if real_only else V_num.subs(substitution_K).n()

        results.append({
            'K0': K_values[0],
            'K1': K_values[1],
            'K2': K_values[2],
            'K3': K_values[3],
            'u': u_value,
            'V': V_value
        })

    # Sort results by potential value.
    results = sorted(results, key=lambda row: row['V'])

    if verbose:
        print("Real solutions found:", len(results))
        if results:
            print("Lowest value of V:", results[0]['V'])

    return results, G


# ------------------------------------------------------------
# 8. Pretty printing
# ------------------------------------------------------------
def print_results(results, max_rows=None):
    """
    Print stationary points in a readable format.

    Parameters
    ----------
    results : list of dict
        Output produced by solve_lightlike_extrema(...).

    max_rows : int or None
        Maximum number of solutions to print. If None, print all.
    """
    if not results:
        print("No real stationary points were found.")
        return

    rows = results if max_rows is None else results[:max_rows]

    for i, row in enumerate(rows, start=1):
        print(f"Solution #{i}")
        print(f"  K0 = {row['K0']}")
        print(f"  K1 = {row['K1']}")
        print(f"  K2 = {row['K2']}")
        print(f"  K3 = {row['K3']}")
        print(f"  u  = {row['u']}")
        print(f"  V  = {row['V']}")
        print("-"*50)


# ------------------------------------------------------------
# 9. Parameter dictionaries
# ------------------------------------------------------------
def default_parameter_point():
    """
    Return one default parameter point for testing the script.

    This example follows the values used in the previous notebook.
    Users should replace these values with the physical parameters
    they want to study.
    """
    return {
        l1: QQ(0),
        l2: QQ(0),
        l3: QQ(1)/10,
        l4: QQ(2)/10,
        l5: QQ(4)/10,
        l6: QQ(4)/10,
        l7: QQ(0),
        v1: QQ(30),
        v2: QQ(171),
        xi: QQ(0)
    }


def default_scan_base_parameters():
    """
    Return the base parameters used for scanning l1 = l2.

    The parameters l1 and l2 are intentionally omitted because they
    are assigned inside scan_lambda_equal(...).
    """
    return {
        l3: QQ(1)/10,
        l4: QQ(2)/10,
        l5: QQ(4)/10,
        l6: QQ(4)/10,
        l7: QQ(0),
        v1: QQ(30),
        v2: QQ(171),
        xi: QQ(0)
    }


# ------------------------------------------------------------
# 10. Parameter scan helper
# ------------------------------------------------------------
def scan_lambda_equal(lambda_values, base_params=None, verbose=False):
    """
    Scan a list of values with l1 = l2 = lambda_value.

    Parameters
    ----------
    lambda_values : iterable
        Values assigned simultaneously to l1 and l2.

    base_params : dict or None
        Dictionary containing the remaining parameters:
            l3, l4, l5, l6, l7, v1, v2, xi
        If None, default values are used.

    verbose : bool
        If True, print the progress of each scan point.

    Returns
    -------
    scan_data : list of dict
        One dictionary per lambda value. Each dictionary contains:
            lambda, results, V_values, V_min, num_solutions
    """
    if base_params is None:
        base_params = default_scan_base_parameters()

    scan_data = []

    for lam in lambda_values:
        params = dict(base_params)
        params[l1] = lam
        params[l2] = lam

        if verbose:
            print(f"Scanning lambda = {lam}")

        results, G = solve_lightlike_extrema(params, verbose=False)
        V_values = [row['V'] for row in results]
        V_min = min(V_values) if V_values else None

        scan_data.append({
            'lambda': lam,
            'results': results,
            'V_values': V_values,
            'V_min': V_min,
            'num_solutions': len(results)
        })

    return scan_data


# ------------------------------------------------------------
# 11. Continuity ordering for plotting solution branches
# ------------------------------------------------------------
def order_branches_by_continuity(list_of_value_lists):
    """
    Reorder solution values to make branch plots more continuous.

    Groebner basis solutions may appear in a different order at each
    scan point. This function chooses the permutation of each new list
    that minimizes the distance from the previous ordered list.

    Parameters
    ----------
    list_of_value_lists : list of lists
        Each inner list contains the V-values found at one scan point.

    Returns
    -------
    ordered : list of lists
        Reordered lists of V-values.
    """
    if not list_of_value_lists:
        return []

    ordered = [sorted(list_of_value_lists[0])]

    for current_values in list_of_value_lists[1:]:
        previous_values = ordered[-1]
        current_values = list(current_values)

        if len(previous_values) != len(current_values):
            ordered.append(sorted(current_values))
            continue

        best_permutation = min(
            itertools.permutations(current_values),
            key=lambda perm: sum(abs(perm[i] - previous_values[i]) for i in range(len(perm)))
        )
        ordered.append(list(best_permutation))

    return ordered


# ------------------------------------------------------------
# 12. Optional plotting and scan-export functions
# ------------------------------------------------------------
def plot_scan_results(
    lambda_values,
    scan_data,
    plot_branches=True,
    plot_minimum=True,
    save_path=None,
    show_plot=True,
    dpi=300
):
    """
    Plot the potential values obtained from a lambda scan.

    This function is optional. It does not affect the Groebner basis
    computation. It only visualizes the numerical output produced by
    scan_lambda_equal(...).

    Parameters
    ----------
    lambda_values : iterable
        Values used in the scan. Usually these are the values assigned to:

            l1 = l2 = lambda

    scan_data : list of dict
        Output produced by scan_lambda_equal(...). Each element must contain:

            'V_values' : list of potential values found at that scan point
            'V_min'    : minimum potential value at that scan point

    plot_branches : bool
        If True, plot all real stationary branches found in the scan.
        Since Groebner solutions can appear in a different order at each
        scan point, the helper order_branches_by_continuity(...) is used
        to make the curves easier to read.

    plot_minimum : bool
        If True, plot the minimum value of the potential at each scan point.
        This is useful when the main interest is the lowest stationary value.

    save_path : str or None
        If a string is given, the figure is saved to that path.
        Example:

            save_path = "lambda_scan.png"

        If None, the figure is not saved.

    show_plot : bool
        If True, display the figure on screen. If you are running the script
        on a remote terminal without graphical output, set this to False and
        use save_path instead.

    dpi : int
        Resolution used when saving the figure.

    Returns
    -------
    fig, ax : matplotlib objects or (None, None)
        The figure and axis objects. If matplotlib is not installed, the
        function prints a message and returns (None, None).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Plotting is not available.")
        print("Install it with one of these options:")
        print("  conda install -c conda-forge matplotlib")
        print("  or")
        print("  sage -pip install matplotlib")
        return None, None

    lambda_values = list(lambda_values)

    if len(lambda_values) != len(scan_data):
        raise ValueError("lambda_values and scan_data must have the same length.")

    V_lists = [row['V_values'] for row in scan_data]
    V_lists_ordered = order_branches_by_continuity(V_lists)

    has_branch_data = any(len(values) > 0 for values in V_lists_ordered)
    has_minimum_data = any(row['V_min'] is not None for row in scan_data)

    if not has_branch_data and not has_minimum_data:
        print("No real stationary values were found. Nothing to plot.")
        return None, None

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    # Reference axes. No specific color is imposed; matplotlib defaults are used.
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    # Plot each stationary branch.
    if plot_branches and has_branch_data:
        max_branches = max(len(values) for values in V_lists_ordered)

        for branch in range(max_branches):
            xs = []
            ys = []

            for lam, values in zip(lambda_values, V_lists_ordered):
                if branch < len(values):
                    xs.append(float(lam))
                    ys.append(float(values[branch]))

            if xs:
                ax.plot(xs, ys, marker='o', linestyle='None', markersize=3,
                        label=f'Stationary branch {branch + 1}')

    # Plot only the minimum value at each scan point.
    if plot_minimum and has_minimum_data:
        xs_min = []
        ys_min = []

        for lam, row in zip(lambda_values, scan_data):
            if row['V_min'] is not None:
                xs_min.append(float(lam))
                ys_min.append(float(row['V_min']))

        if xs_min:
            ax.plot(xs_min, ys_min, marker='s', linewidth=1.5,
                    label='Minimum V')

    ax.set_xlabel(r'$\lambda_1 = \lambda_2$')
    ax.set_ylabel(r'$V$')
    ax.set_title('Lightlike stationary values from Groebner-basis scan')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def export_scan_results_csv(scan_data, output_path):
    """
    Export the lambda scan summary to a CSV file.

    This helper is optional but useful when you want to analyze the scan
    results later in Excel, Python, or another plotting tool.

    Parameters
    ----------
    scan_data : list of dict
        Output produced by scan_lambda_equal(...).

    output_path : str
        Path of the CSV file to create.

    CSV columns
    -----------
    lambda : scanned value assigned to l1 and l2
    V_min : minimum stationary value found at that lambda
    num_solutions : number of real stationary solutions found
    V_values : all stationary values joined with semicolons
    """
    import csv

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['lambda', 'V_min', 'num_solutions', 'V_values'])

        for row in scan_data:
            V_values_as_text = ';'.join(str(v) for v in row['V_values'])
            writer.writerow([
                row['lambda'],
                row['V_min'],
                row['num_solutions'],
                V_values_as_text
            ])

    print(f"CSV file saved to: {output_path}")


def run_lambda_scan_and_plot(
    lambda_values=None,
    base_params=None,
    save_plot_path='lambda_scan.png',
    save_csv_path='lambda_scan_results.csv',
    show_plot=False,
    verbose=True
):
    """
    Run the lambda scan and optionally save both the plot and the CSV summary.

    This is the easiest function to use if you want the optional plotting
    workflow in one command.

    Parameters
    ----------
    lambda_values : iterable or None
        Values assigned to l1 = l2. If None, a default interval is used:

            -0.05, -0.048, ..., 0.05

    base_params : dict or None
        Dictionary with the remaining model parameters:

            l3, l4, l5, l6, l7, v1, v2, xi

        If None, default_scan_base_parameters() is used.

    save_plot_path : str or None
        Path where the plot will be saved. If None, no plot is saved.

    save_csv_path : str or None
        Path where the CSV summary will be saved. If None, no CSV is saved.

    show_plot : bool
        If True, display the plot interactively. For terminal execution,
        False is usually safer.

    verbose : bool
        If True, print scan progress.

    Returns
    -------
    scan_data : list of dict
        Full output of scan_lambda_equal(...).
    """
    if lambda_values is None:
        lambda_values = [QQ(-5)/100 + i*QQ(1)/500 for i in range(51)]
    else:
        lambda_values = list(lambda_values)

    scan_data = scan_lambda_equal(lambda_values, base_params=base_params, verbose=verbose)

    if save_csv_path is not None:
        export_scan_results_csv(scan_data, save_csv_path)

    if save_plot_path is not None or show_plot:
        plot_scan_results(
            lambda_values,
            scan_data,
            plot_branches=True,
            plot_minimum=True,
            save_path=save_plot_path,
            show_plot=show_plot
        )

    return scan_data


# ------------------------------------------------------------
# 13. Main execution block
# ------------------------------------------------------------
def main():
    """
    Main execution block.

    By default, this runs only one fixed parameter example.
    The scan is provided but disabled to avoid long runtimes.
    """
    print("Two-Higgs-doublet model in bilinear variables K^mu")
    print("="*60)
    print_required_parameters()

    # --------------------------------------------------------
    # Example 1: solve one fixed parameter point.
    # --------------------------------------------------------
    print("\nRunning one example parameter point...")
    params = default_parameter_point()
    results, G = solve_lightlike_extrema(params, verbose=True)
    print_results(results)

    # --------------------------------------------------------
    # Example 2: optional scan.
    # Set RUN_SCAN = True if you want to perform the scan.
    # --------------------------------------------------------
    RUN_SCAN = False

    if RUN_SCAN:
        print("\nRunning lambda scan with l1 = l2...")
        # This function performs the scan, saves a CSV summary,
        # and saves a PNG plot. It does not require an interactive window.
        scan_data = run_lambda_scan_and_plot(
            save_plot_path='lambda_scan.png',
            save_csv_path='lambda_scan_results.csv',
            show_plot=False,
            verbose=True
        )


# This block runs only when the file is executed directly with SageMath.
# It does not run automatically if the file is loaded from another script.
if __name__ == "__main__":
    main()
