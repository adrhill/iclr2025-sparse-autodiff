---
layout: distill
title: An Illustrated Guide to Sparse Automatic Differentiation
description: Your blog post's abstract.
  Please add your abstract or summary here and not in the main body of your text. 
  Do not include math/latex or hyperlinks.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Build page via 
# ```bash
# rbenv init
# rbenv install 3.3.0
# rbenv local 3.3.0
# bundle install
# bundle exec jekyll serve --future --open-url /blog/sparse-autodiff/ --livereload
# ```
#
# Then navigate to `/blog/sparse-autodiff/`

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Adrian Hill
    url: "http://adrianhill.de/"
    affiliations:
      name: Machine Learning Group, TU Berlin
  - name: Guillaume Dalle
    url: "https://gdalle.github.io"
    affiliations:
      name: IdePHICS, INDY and SPOC laboratories, EPFL

# must be the exact same name as your blogpost
bibliography: 2025-04-28-sparse-ad.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Automatic differentiation
    subsections:
    - name: Toy example
    - name: The chain rule
    - name: Automatic differentiation is matrix-free
    - name: Evaluating linear maps
    - name: Forward-mode AD
    - name: Reverse-mode AD
  - name: Sparse AD
  - name: Sparsity pattern detection
  - name: Matrix coloring
  - name: Second-order sparse AD
  - name: Demonstration

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: ##bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

While the use of gradient-based optimization is ubiquitous in machine learning,
the usage of Jacobians and second-order-optimization via Hessians remains scarce.
This often motivated by the high computational cost of these matrices.
However, in numerous applications within scientific machine learning, 
Jacobians and Hessians exhibit sparsity, a characteristic that–when leveraged–has the potential to vastly accelerate computation.
While the use of **Automatic Differentiation** (AD) via frameworks and programming languages like PyTorch, JAX and Julia is ubiquitous, **sparse AD** is mostly unknown.

With this blog post, we aim to shed light on the inner workings of sparse AD, 
starting out with a high-level introduction into classical AD, 
covering the computation of Jacobians in both forward- and reverse-mode.
We then dive into the two primary components of sparse AD:
sparsity pattern **detection** and **coloring**.
Having covered the computation of sparse Jacobians, 
we then move on to sparse Hessians.  
We conclude with a demonstration of sparse automatic differentiation,
providing performance benchmarks and guidance on when to use sparse AD over "dense" AD.

## Automatic differentiation

We start out by covering the fundamentals of classic AD, which we will refer to as "dense" AD, in distinction to sparse AD.

### Toy example

AD makes use of the **compositional structure** of mathematical functions like deep neural networks.
As our motivating example, we will therefore take a look at a differentiable function $f$
composed from differentiable $g: \mathbb{R}^{n} \rightarrow \mathbb{R}^{p}$ 
and $h: \mathbb{R}^{p} \rightarrow \mathbb{R}^{m}$, 
such that $f = h \circ g: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$.
The insights gained from this toy example should translate directly to more deeply composed functions.

### The chain rule

For a function $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ and a point of linearization $\mathbf{x} \in \mathbb{R}^{n}$, 
the Jacobian $J_f(\mathbf{x})$ is the $m \times n$ matrix of first-order partial derivatives, such that the $(i,j)$-th entry is

$$ (J_f(\mathbf{x}))_{i,j} = \frac{\partial f_i}{\partial x_j}(\mathbf{x}) \in \mathbb{R} \quad . $$

When viewed as a  linear map, this Jacobian can be though of as the **linear approximation** of $f$ around $\mathbf{x}$.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/chainrule_num.svg" class="img-fluid" %}
<div class="caption">
    Figure 1: Visualization of the multivariate chain rule for $f = h \circ g$.
</div>

For a composed function $f = h \circ g$, the **multivariable chain rule** tells us that we obtain the Jacobian of $f$ by **composing** the Jacobians of $h$ and $g$:

$$ J_f(\mathbf{x}) = J_{h \circ g}(\mathbf{x}) =J_h(g(\mathbf{x})) \cdot J_g(\mathbf{x}) \quad .$$

Figure 1 illustrates this for $n=5$, $m=4$ and $p=3$.
<!-- TODO: explain that values are random? -->
Without loss of generality, we will keep using these dimensionalities in following illustrations.

### Automatic differentiation is matrix-free

We've seen how the chain rule directly translates the compositional structure of a function into the compositional structure of its Jacobian.
Due to the small size of our chosen dimensions $n$, $m$ and $p$, this approach worked well on our toy example in Figure 1.  
In practice however, there is a problem:
**Keeping intermediate Jacobian matrices in computer memory is inefficient and often impossible.**

We will refer to this kind of matrix, for which all entries are kept in computer memory, as a **materialized**.
Examples for materialized matrices include NumPy's `ndarray`, PyTorch's `Tensor`s, JAX's `Array` and Julia's `Matrix`.
<!-- TODO: Check capitalization of Python types. It's the wild west over there. -->

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/big_conv_jacobian.png" class="img-fluid" %}
<div class="caption">
    Figure 2: Structure of the Jacobian of a tiny convolutional layer.
</div>


<!-- TODO: Maybe the identity function is a simpler example? -->
As a motivating example against **materialized Jacobians**, let's take a look at a tiny convolutional layer.
We assume a convolutional filter of size $5 \times 5$, as well as a single input and a single output channel.
An input of size $28 \times 28 \times 1$ results in a $576 \times 784$ Jacobian, the structure of which is shown in Figure 2.
Computing it would be highly memory inefficient, as $96.8\%$ of all entries are zero.
Additionally, matrix multiplication with the Jacobians of following layers would be computationally inefficient due to numerous redundant additions and multiplications by zero.

In modern neural network architectures, which are crossing the threshold of one trillion parameters, 
computing intermediate Jacobians is not only inefficient, but also exceeds available memory.
Further examples include the Jacobians resulting from an identity function or any activation function that is applied element-wise.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/chainrule_num.svg" class="img-fluid" %}
<div class="caption">
    Figure 3a: Chain rule using materialized Jacobians (solid outline).
</div>

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/matrixfree.svg" class="img-fluid" %}
<div class="caption">
    Figure 3b: Chain rule using matrix-free linear maps (dashed outline).
</div>

Since keeping **materialized** Jacobian matrices in memory is inefficient or impossible,
AD instead implements **linear maps**, **functions** that act exactly like materialized matrices.

<!-- TODO: "In terms  of notation" or "Mathematically speaking"? -->
In terms of notation, this linear map can be obtained by applying the differential operator $D$ to $f$. 
The resulting function $Df(\mathbf{x})$ corresponds to the linear approximation of $f$ at $\mathbf{x}$.
We can rephrase  the chain rule as   

$$ Df(\mathbf{x}) = D(h \circ g)(\mathbf{x}) =Dh(g(\mathbf{x})) \circ Dg(\mathbf{x}) \quad .$$

Note that all terms in this formulation of the chain rule are functions.
A visualization for our toy example can be found in Figure 3b. 
Our illustrations distinguish between materialized matrices and linear maps by using solid and dashed lines respectively.


*We visualize "matrix entries" in linear maps to build intuition.
Even though following illustrations will sometimes put numbers onto these "matrix entries", 
linear maps are best thought of as black-box functions.*

### Evaluating linear maps

Efficiently **materializing** these functions to a matrix $J_f$ is what this talk is about.

We only propagate **materialized vectors** (*solid*) through our **linear maps** (*dashed*):

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/matrixfree2.svg" class="img-fluid" %}
<div class="caption">
    Figure 4: Evaluating linear maps in forward-mode.
</div>


### Forward-mode AD

**Materialize $J$ column-wise**: number of evaluations matches **input dimensionality**

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode.svg" class="img-fluid" %}
<div class="caption">
    Figure 5: Forward-mode AD materializes Jacobians column-by-column.
</div>

This is called a **Jacobian-vector product** (JVP) or **pushforward**.

<!-- * I personally prefer **pushforward**, since the JVP could imply a materialized matrix.
* Note that while this might look redundant at first glance, it took a **linear map** (*dashed*) and turned it into a **materialized matrix** (*solid*)  -->

### Reverse-mode AD

**Materialize $J$ row-wise**: number of evaluations matches **output dimensionality**

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/reverse_mode.svg" class="img-fluid" %}
<div class="caption">
    Figure 6: Reverse-mode AD materializes Jacobians row-by-row.
</div>

This is called a **vector-Jacobian product** (VJP) or **pullback**.

### Special case: *"Backpropagation"*
The gradient of a scalar function $f : \mathbb{R}^n \rightarrow \mathbb{R}$ requires just **one** evaluation with $\mathbf{e}_1=1$.

## Sparse AD

### Sparsity

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_matrix.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_map.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 7: A sparse matrix and its respective sparse linear map.
</div>

A matrix in which most elements are zero.

A linear map that materializes to a sparse matrix.

<!-- ::: {.callout-note}
### Remark: Sparsity of computer programs
Compute graphs of programs are almost always "dense": the existence of superfluous operations could be considered a bug. 
However, corresponding Jacobians can still be sparse. As an example, consider a convolution.
::: -->

### Core Idea: Exploit structure

**Assuming the structure of the Jacobian is known, we can materialize several columns of the Jacobian in a single evaluation:**

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_ad.svg" class="img-fluid" %}
<div class="caption">
    Figure 8: Materializing multiple orthogonal columns of a Jacobian in forward-mode.
</div>

* Linear maps are **additive**: $\;Df(e_i+\ldots+e_j) = Df(e_i) +\ldots+ Df(e_j)$
* The RHS summands are columns of the Jacobian
* If columns are **orthogonal** and their **structure is known**, the sum can be decomposed 

The same idea also applies to rows in reverse-mode.

### Problem: The structure of the Jacobian is unknown
* The linear map is a black-box function
* **Without materializing the linear map, the structure of the Jacobian is unknown**
* If we fully materialize the Jacobian via "dense AD", sparse AD isn't needed
:::

<!-- * **"Coloring":** find orthogonal columns (or rows) via graph coloring  -->

### The Solution: Sparsity patterns

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparsity_pattern.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/coloring.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 9: The two elementary steps in sparse AD: (a) sparsity pattern detection, (b) coloring of the sparsity pattern.
</div>

Performance is the crux of Sparse AD
* These two steps need to be faster than the computation of columns/rows they allow us to skip. Otherwise, we didn't gain any performance...
* ...unless we are able to reuse the pattern!

<!-- * **"Coloring":** find orthogonal columns (or rows) via graph coloring  -->

## Sparsity pattern detection

### Index sets

Binary Jacobian patterns are efficiently compressed using **indices of non-zero values**:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_matrix.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparsity_pattern.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparsity_pattern_compressed.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 10: Equivalent sparsity pattern representations: (a) uncompressed matrix, (b) binary pattern, (c) index set (compressed along rows).
</div>

(Since the method we are about to show is essentially a binary forward-mode AD system, we compress along rows.)


### Core Idea: Propagate index sets

**Naive approach:** materialize full Jacobians (inefficient or impossible):

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode_naive.svg" class="img-fluid" %}
<div class="caption">
    Figure 11: Materializing a Jacobian forward-mode. 
    Due to high memory requirements for intermediate Jacobians, this approach is inefficient or impossible.  
</div>

**Our goal:** propagate full basis index sets:

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode_sparse.svg" class="img-fluid" %}
<div class="caption">
    Figure 12: Propagating an index set through a linear map to obtain a sparsity pattern.  
</div>

**But how do we define these propagation rules?**

## Matrix coloring

## Second-order sparse AD

## Demonstration
