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
# bundle exec jekyll serve --future
# ```

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
    - name: Chain rule
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

Note: please use the table of contents as defined in the front matter rather than the traditional markdown styling.

## Automatic differentiation

### Chain rule

For a **composed function** $f(x) = h(g(x))$, 
the chain rule tells us that we obtain the Jacobian of $f$ by **composing the Jacobians** of $h$ and $g$:

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/chainrule_num.svg" class="img-fluid" %}


where the $(i,j)$-th entry in the Jacobian is $(J_f(x))_{i,j} = \frac{\partial f_i}{\partial x_j}(x) \in \mathbb{R}$."

### Problem: Jacobians are too large

Keeping intermediate Jacobian **matrices in memory** is inefficient or even impossible.


{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/big_conv_jacobian.png" class="img-fluid" %}


**Example:** Tiny convolutional layer

* $5 \times 5$ filter, $1$ input channel, $1$ output channel
* Input size: $28 \times 28 \times 1$ 
* Resulting Jacobian:
  * size $576 \times 784$
  * $96.8\%$ of entries are zero

### Definition: Materialized matrices
By **materialized matrices**, we refer to matrices $A$ for which entries $(A)_{i,j}$ are kept in **computer memory**,
e.g. a NumPy `np.array` or Julia `Matrix`.

### AD is matrix-free

Keeping full Jacobian **matrices in memory** (*solid*) is inefficient or even impossible.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/chainrule_num.svg" class="img-fluid" %}

Instead, AD implements **functions** (*dashed*) that act exactly like Jacobians.
These are **linear maps**. 
Denoted using the differential operator $D$.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/matrixfree.svg" class="img-fluid" %}

Efficiently **materializing** these functions to a matrix $J_f$ is what this talk is about! 

<!-- * Figures:
  * materialized matrices: solid border
  * non-materialized matrices: dashed border
* Even though I will sometimes put numbers into linear maps in the upcoming slides, they should be considered "**opaque black boxes**" until materialized into matrices
* Intuitive example for why materialization can be bad: **identity function** -->

### Evaluating linear maps

We only propagate **materialized vectors** (*solid*) through our **linear maps** (*dashed*):

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/matrixfree2.svg" class="img-fluid" %}

### Forward-mode AD

**Materialize $J$ column-wise**: number of evaluations matches **input dimensionality**

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode.svg" class="img-fluid" %}

This is called a **Jacobian-vector product** (JVP) or **pushforward**.

<!-- * I personally prefer **pushforward**, since the JVP could imply a materialized matrix.
* Note that while this might look redundant at first glance, it took a **linear map** (*dashed*) and turned it into a **materialized matrix** (*solid*)  -->

### Reverse-mode AD

**Materialize $J$ row-wise**: number of evaluations matches **output dimensionality**

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/reverse_mode.svg" class="img-fluid" %}

This is called a **vector-Jacobian product** (VJP) or **pullback**.

### Special case: *"Backpropagation"*
The gradient of a scalar function $f : \mathbb{R}^n \rightarrow \mathbb{R}$ requires just **one** evaluation with $\mathbf{e}_1=1$.

## Sparse AD

### Sparsity

**Sparse Matrix**

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_matrix.svg" class="img-fluid" %}

A matrix in which most elements are zero.


**Sparse Linear Map**

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_map.svg" class="img-fluid" %}

A linear map that materializes to a sparse matrix.


<!-- ::: {.callout-note}
### Remark: Sparsity of computer programs
Compute graphs of programs are almost always "dense": the existence of superfluous operations could be considered a bug. 
However, corresponding Jacobians can still be sparse. As an example, consider a convolution.
::: -->

### Core Idea: Exploit structure

**Assuming the structure of the Jacobian is known, we can materialize several columns of the Jacobian in a single evaluation:**

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_ad.svg" class="img-fluid" %}


* Linear maps are **additive**: $\;Df(e_i+\ldots+e_j) = Df(e_i) +\ldots+ Df(e_j)$
* The RHS summands are columns of the Jacobian
* If columns are **orthogonal** and their **structure is known**, the sum can be decomposed 

The same idea also applies to rows in reverse-mode.

### But there is a problem...

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_map_colored.svg" class="img-fluid" %}

### Unfortunately, the structure of the Jacobian is unknown
* The linear map is a black-box function
* **Without materializing the linear map, the structure of the Jacobian is unknown**
* If we fully materialize the Jacobian via "dense AD", sparse AD isn't needed
:::

<!-- * **"Coloring":** find orthogonal columns (or rows) via graph coloring  -->

### The Solution: Sparsity patterns

**Step 1:** Pattern detection

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparsity_pattern.svg" class="img-fluid" %}

**Step 2:** Pattern coloring

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/coloring.svg" class="img-fluid" %}

### Performance is the crux of Sparse AD
* These two steps need to be faster than the computation of columns/rows they allow us to skip. Otherwise, we didn't gain any performance...
* ...unless we are able to reuse the pattern!

<!-- * **"Coloring":** find orthogonal columns (or rows) via graph coloring  -->

## Sparsity pattern detection

### Index sets

Binary Jacobian patterns are efficiently compressed using **indices of non-zero values**:

**Uncompressed**

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_matrix.svg" class="img-fluid" %}

**Binary Pattern**

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparsity_pattern.svg" class="img-fluid" %}

**Index Set**

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparsity_pattern_compressed.svg" class="img-fluid" %}


(Since the method we are about to show is essentially a binary forward-mode AD system, we compress along rows.)


### Core Idea: Propagate index sets

**Naive approach:** materialize full Jacobians (inefficient or impossible):

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode_naive.svg" class="img-fluid" %}

**Our goal:** propagate full basis index sets:

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode_sparse.svg" class="img-fluid" %}

**But how do we define these propagation rules?**
Let's do some analysis!

## Matrix coloring

## Second-order sparse AD

## Demonstration
