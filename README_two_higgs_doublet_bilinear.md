# Two-Higgs-Doublet Potential in Bilinear Variables

This project contains an improved SageMath implementation for studying the scalar potential of the **two-Higgs-doublet model (2HDM)** using the **bilinear formalism** with variables:

\[
K^\mu = (K_0, K_1, K_2, K_3)
\]

The code builds the 2HDM potential, applies the light-cone constraint, solves the stationary equations using Gröbner bases, filters real solutions, evaluates the potential at each stationary point, and performs parameter scans.

---

## 1. Main idea

The potential is written in bilinear variables as:

\[
V(K)=\xi_0K_0+\vec{\xi}\cdot\vec{K}+\eta_{00}K_0^2+2\eta K_0K_3+\vec{K}^{T}E\vec{K}
\]

where:

\[
\vec{K}=(K_1,K_2,K_3)^T
\]

The lightlike condition used in the code is:

\[
-K_0^2+K_1^2+K_2^2+K_3^2=0
\]

This condition is imposed through a Lagrange multiplier `u`.

---

## 2. Files

### `two_higgs_doublet_bilinear_improved.sage`

Main SageMath script. It contains all functions and example executions.

### `two_higgs_doublet_bilinear_improved.ipynb`

Notebook version of the same code, useful for SageMath notebooks, CoCalc, or a Jupyter environment with SageMath support.

---

## 3. Requirements

This code is written for **SageMath**.

### Installing SageMath

On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install sagemath
```

On Fedora:

```bash
sudo dnf install sagemath
```

On Arch Linux:

```bash
sudo pacman -S sagemath
```

On macOS with Homebrew:

```bash
brew install --cask sage
```

On Windows, the recommended local option is to install SageMath through WSL.
For example, after installing Ubuntu in WSL:

```bash
sudo apt update
sudo apt install sagemath
```

To verify the installation:

```bash
sage --version
```

Recommended environments:

- SageMath installed locally.
- CoCalc with SageMath kernel.
- SageMathCell for small tests.
- Jupyter Notebook with SageMath kernel.

Python alone is not enough because the code uses Sage-specific objects such as:

```python
var()
QQ
QQbar
RR
CC
PolynomialRing()
ideal()
groebner_basis()
```

For plotting, the code also uses:

```python
matplotlib
```

---

## 4. Model parameters

The code uses the following model parameters:

```python
l1, l2, l3, l4, l5, l6, l7
v1, v2, xi
```

They are stored globally as:

```python
MODEL_PARAMS = (l1, l2, l3, l4, l5, l6, l7, v1, v2, xi)
```

A numerical parameter dictionary must provide values for all of them.

Example:

```python
params_example = {
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
```

It is recommended to use exact rational values like:

```python
QQ(1)/10
```

instead of decimal floats like:

```python
0.1
```

Exact values make Gröbner basis computations more stable.

---

## 5. Main functions

### `scalar_part(expr)`

Extracts the scalar entry from a `1x1` Sage matrix.

This is useful because expressions such as:

```python
row_vector * matrix * column_vector
```

usually return a `1x1` matrix in Sage, not a direct scalar expression.

---

### `is_almost_real(z, tol=1e-10)`

Checks whether a numerical algebraic value is real within a given tolerance.

This is useful because solutions obtained over `QQbar` can contain very small imaginary numerical residues.

---

### `real_part(z)`

Returns the real part of a Sage numerical value.

---

### `validate_parameter_dictionary(params)`

Checks whether all required model parameters are present in the parameter dictionary.

It also warns if Python floats are used, because exact rational values are safer for Gröbner basis computations.

---

### `build_2hdm_bilinear_potential()`

Builds the symbolic 2HDM potential in bilinear variables.

Returns:

```python
V, lightcone_constraint
```

where:

```python
V
```

is the symbolic scalar potential, and:

```python
lightcone_constraint
```

is:

\[
-K_0^2+K_1^2+K_2^2+K_3^2
\]

---

### `build_lagrange_system(params)`

Builds the Lagrange system for the constrained stationary problem.

The Lagrange function is:

\[
L(K,u)=V(K)+u\,g(K)
\]

where:

\[
g(K)=-K_0^2+K_1^2+K_2^2+K_3^2
\]

The resulting equations are:

\[
\frac{\partial L}{\partial K_0}=0,
\quad
\frac{\partial L}{\partial K_1}=0,
\quad
\frac{\partial L}{\partial K_2}=0,
\quad
\frac{\partial L}{\partial K_3}=0,
\quad
g(K)=0
\]

Returns:

```python
equations, V_num, constraint
```

---

### `solve_lightlike_extrema(params, field=QQbar, real_only=True, tol=1e-10, verbose=True)`

Solves the lightlike stationary points using Gröbner bases.

Main steps:

1. Builds the Lagrange equations.
2. Converts them into a polynomial ring.
3. Constructs the ideal.
4. Computes the Gröbner basis.
5. Solves the zero-dimensional ideal using `variety(QQbar)`.
6. Filters real solutions.
7. Evaluates the potential at each stationary point.
8. Sorts solutions from lowest to highest value of `V`.

Returns:

```python
results, G
```

where:

- `results` is a list of dictionaries containing `K0`, `K1`, `K2`, `K3`, `u`, and `V`.
- `G` is the Gröbner basis.

Example:

```python
results, G = solve_lightlike_extrema(params_example, verbose=True)
```

---

### `print_results(results, max_rows=None)`

Prints the stationary points in a readable format.

Example:

```python
print_results(results)
```

To print only the first three solutions:

```python
print_results(results, max_rows=3)
```

---

### `scan_lambda_equal(lambda_values, base_params=None, verbose=False)`

Scans the model imposing:

\[
\lambda_1=\lambda_2
\]

For each value, the function solves the lightlike extrema and stores the potential values.

Example:

```python
lambda_values = [QQ(-5)/100 + i*QQ(1)/500 for i in range(51)]
scan_data = scan_lambda_equal(lambda_values, verbose=True)
```

Each element of `scan_data` contains:

```python
{
    'lambda': lam,
    'results': results,
    'V_values': V_values,
    'V_min': V_min,
    'num_solutions': len(results)
}
```

---

### `order_branches_by_continuity(list_of_value_lists)`

Reorders the solution branches during a parameter scan.

This is necessary because solutions obtained from Gröbner bases may appear in different orders at different scan points.

The function compares each new list of potential values with the previous one and chooses the ordering that minimizes the total distance between branches.

---

## 6. How to run the code

### Step 1: Open the SageMath environment

Use one of these options:

- SageMath notebook.
- CoCalc SageMath kernel.
- Local SageMath terminal.
- Jupyter Notebook with SageMath kernel.

### Step 2: Load or paste the code

If using the `.sage` file, run:

```sage
load("two_higgs_doublet_bilinear_improved.sage")
```

If using the notebook, execute the cells in order.

### Step 3: Define parameter values

Example:

```python
params_example = {
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
```

### Step 4: Solve the lightlike extrema

```python
results, G = solve_lightlike_extrema(params_example, verbose=True)
print_results(results)
```

### Step 5: Scan values of `l1 = l2`

```python
lambda_values = [QQ(-5)/100 + i*QQ(1)/500 for i in range(51)]
scan_data = scan_lambda_equal(lambda_values, verbose=True)
```

### Step 6: Plot branches

```python
V_lists = [row['V_values'] for row in scan_data]
V_lists_ordered = order_branches_by_continuity(V_lists)
```

Then use the plotting block included at the end of the script.

---

## 7. Expected output

For a single parameter point, the code prints:

- Number of elements in the Gröbner basis.
- Dimension of the ideal.
- Number of real solutions found.
- Lowest value of the potential.
- List of stationary points.

Example output structure:

```text
Groebner basis computed.
Number of basis elements: ...
Ideal dimension: 0
Real solutions found: ...
Lowest value of V: ...
```

Each solution is printed as:

```text
Solution #1
  K0 = ...
  K1 = ...
  K2 = ...
  K3 = ...
  u  = ...
  V  = ...
--------------------------------------------------
```

---

## 8. Notes about exact arithmetic

For Gröbner bases, exact arithmetic is strongly preferred.

Use:

```python
QQ(1)/10
```

instead of:

```python
0.1
```

This avoids numerical instability and makes the polynomial ring construction more reliable.

If the parameter `xi` produces trigonometric values that are not rational, it may be necessary to work over `QQbar` or to substitute exact algebraic values whenever possible.

---

## 9. Current scope

This version focuses on the **lightlike sector**:

\[
-K_0^2+K_1^2+K_2^2+K_3^2=0
\]

It does not yet implement the full stability algorithm separated into regions such as:

- timelike region,
- lightlike region,
- tip of the cone,
- hierarchical tests such as `J4`, `J3`, `J2`, and `J1`.

Those blocks can be added later on top of the current modular structure.

---

## 10. Suggested next improvements

Possible next steps:

1. Add a function to solve the timelike sector.
2. Add a function to evaluate the tip of the cone.
3. Organize the stability algorithm into `J4`, `J3`, `J2`, and `J1`.
4. Generalize the code from the pure 2HDM case to two Higgs doublets plus `n` real singlets.
5. Export scan results to CSV.
6. Add automatic plots for all solution branches.
7. Add exception handling when the ideal is not zero-dimensional.

---

## 11. Minimal example

```python
params_example = {
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

results, G = solve_lightlike_extrema(params_example, verbose=True)
print_results(results)
```

---

## 12. Author note

This code is designed as a cleaner and more modular version of the original SageMath notebook. The goal is to make the implementation easier to extend, especially for future work involving Gröbner bases, stability regions, and scalar potentials with additional singlet fields.
