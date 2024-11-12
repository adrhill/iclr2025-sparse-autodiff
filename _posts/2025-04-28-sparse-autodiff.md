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
    - name: The chain rule
    - name: AD is matrix-free
    - name: Forward-mode AD
    - name: Reverse-mode AD
  - name: Sparse automatic differentiation
    subsections:
    - name: Sparse matrices
    - name: Leveraging structure
    - name: Sparsity pattern detection and coloring
  - name: Pattern detection
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
the usage of Jacobians and second-order-optimization via Hessians remains scarce due to high computational costs.
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

### AD is matrix-free

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

*TODO: replace with identity function on arrays as a simpler example?*

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

### Forward-mode AD

Now that we have translated the compositional structure of our function $f$ into a compositional structure of linear maps, we can evaluate them by propagating **materialized vectors** through them.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/matrixfree2.svg" class="img-fluid" %}
<div class="caption">
    Figure 4: Evaluating linear maps in forward-mode.
</div>

Figure 4 illustrates the propagation of a vector $\mathbf{v}_1 \in \mathbb{R}^n$ from the right-hand side.
Since we propagate in the order of the original function evaluation, this is called **forward-mode AD**.

In the first step, we evaluate $Dg(\mathbf{x})(\mathbf{v}_1)$.
Since this operation by definition corresponds to 

$$ \mathbf{v}_2 = Dg(\mathbf{x})(\mathbf{v}_1) = J_{g}(\mathbf{x}) \cdot \mathbf{v}_1 \;\in \mathbb{R}^p \quad ,$$

it is also commonly called a **Jacobian-vector product** (JVP) or **pushforward**.
The resulting vector $\mathbf{v}_2$ is then used to compute the subsequent JVP 

$$ \mathbf{v}_3 = Dh(g(\mathbf{x}))(\mathbf{v}_2) = J_{h}(g(\mathbf{x})) \cdot \mathbf{v}_2 \;\in \mathbb{R}^m \quad ,$$

which in accordance with the chain rule is equivalent to 

$$ \mathbf{v}_3 = Df(\mathbf{x})(\mathbf{v}_1) = J_{f}(\mathbf{x}) \cdot \mathbf{v}_1 \quad ,$$

the JVP of our composed function $f$.

**Note that we didn't materialize intermediate Jacobians at any point**–we only propagated vectors.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode.svg" class="img-fluid" %}
<div class="caption">
    Figure 5: Forward-mode AD materializes Jacobians column-by-column.
</div>

But how can we use this machinery to compute a **materialized Jacobian**? 
Figure 5 shows the answer.
While it might look redundant at first, evaluating the **linear map** $Df(\mathbf{x})$ with the $i$-th standard basis vector **materializes** the $i$-th column of the Jacobian $J_f(\mathbf{x})$. 
Materializing the full $m \times n$ Jacobian takes $n$ evaluations with all $n$ standard basis vectors,
as many as there are inputs.

When applied to gradient-based optimization of neural networks with large amounts of parameters, 
this dependence on the input dimensionality hinders the performance of forward-mode AD.
Luckily, we can also propagate vectors through our linear maps from the left-hand side, resulting in **reverse-mode AD**.

### Reverse-mode AD

<!-- TODO: add analogous reverse-mode figure -->
*TODO: Add and describe reverse-mode equivalent of figure 4.*

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/reverse_mode.svg" class="img-fluid" %}
<div class="caption">
    Figure 6: Reverse-mode AD materializes Jacobians row-by-row.
</div>

As illustated in Figure 6, we can also **materialize Jacobians row by row**.
Analogous to forward-mode in figure 5, 
this requires evaluating $m$ VJPs with all $m$ standard basis vectors,
as many as there are outputs.

Since neural networks are usually trained using scalar loss functions,
reverse-mode only requires the evaluation of a single VJP to compute a gradient.
This makes it the method of choice for machine learners, 
who more commonly refer to reverse-mode AD as  *backpropagation*.

## Sparse automatic differentiation

### Sparse matrices

Sparse matrices are matrices in which most elements are zero.
We refer to linear maps as "sparse linear maps" if they materialize to sparse matrices.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_matrix.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_map.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 7: A sparse Jacobian and its respective sparse linear map.
</div>

Compute graphs of programs are almost always "dense", 
as the existence of superfluous operations would be considered a bug. 
However, corresponding Jacobians can still be sparse. 
As an example, consider the previous example of an efficiently programmed convolutional layer.
A toy example of a sparse Jacobian matrix and it's respective linear map is shown in Figure 7.

### Leveraging structure

Assuming we know the structure of the Jacobian, we can find orthogonal, 
non-overlapping columns or rows via a method called **matrix coloring** that we will go into more detail on later.

**The core idea of sparse AD is that we can materialize multiple orthogonal columns or rows in a single evaluation.**
Since linear maps are additive, it always holds that

$$ Df(\mathbf{x})(\mathbf{e}_i+\ldots+e_j) = Df(\mathbf{x})(\mathbf{e}_i) +\ldots+ Df(\mathbf{x})(\mathbf{e}_j) \quad .$$

The right hand side summands each correspond to a column of the Jacobian.
If the columns are **orthogonal** and their **structure is known**, 
the sum can be decomposed into its summands, materializing multiple columns in a single JVP.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_ad.svg" class="img-fluid" %}
<div class="caption">
    Figure 8: Materializing multiple orthogonal columns of a Jacobian in forward-mode.
</div>

This specific example using JVPs corresponds to sparse forward-mode AD 
and is visualized in Figure 8, where all orthogonal columns have been colored in matching hues.
By computing a single JVP with the vector $\mathbf{e}_1 + \mathbf{e}_2 + \mathbf{e}_5$, 
we materialize the sum of the first, second and fifth column of our Jacobian.
Since we can assume we know the structure of the Jacobian,
we can assign the values in the resulting vector to the correct Jacobian entries. 

The same idea can also be applied to reverse mode AD.
Instead of finding orthogonal column, we need to find orthogonal rows.
We can then materialize multiple rows in a single VJP.

*TODO: illustrate reverse-mode*
<!-- TODO: illustrate reverse mode -->

### Sparsity pattern detection and coloring

Unfortunately, our initial assumption had a major flaw: 
Since AD only gives us a composition of linear maps and linear maps are black-box functions,
the structure of the Jacobian is completely unknown.

**We can't tell which rows and columns are orthogonal without first materializing a Jacobian matrix.**
But if we fully materialize a Jacobian via dense AD, sparse AD isn't needed.

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

The solution to this problem is shown in Figure 9:
in order to find orthogonal columns (or rows), we don't need to materialize the full Jacobian.
Instead, it is enough to materialize a binary sparsity pattern of the Jacobian.
This pattern contains enough information to color it.

Performance is key: For one-off computations, these two steps need to be faster than the computation of columns or rows they allow us to skip. Otherwise, we didn't gain any performance.
As we will see in later benchmarks, this level of performance can be achieved.
Additionally, if we need to compute Jacobians multiple times and are able to reuse the sparsity pattern, 
the cost of sparsity pattern detection and coloring can be amortized over time.


## Pattern detection

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

### Matrix coloring

## Second-order sparse AD

## Demonstration
