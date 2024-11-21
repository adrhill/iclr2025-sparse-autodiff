---
layout: distill
title: An Illustrated Guide to Automatic Sparse Differentiation
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

# TODO before submission:
# - revert CI workflows
# - check correct figure caption numbering and references
# - check correct rendering of SVGs on multiple browsers

authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-sparse-autodiff.bib  

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
    - name: From linear maps back to Jacobians
  - name: Automatic sparse differentiation
    subsections:
    - name: Sparse matrices
    - name: Leveraging sparsity
    - name: Sparsity pattern detection and coloring
  - name: Pattern detection
    subsections:
    - name: Index sets
    - name: Efficient propagation
    - name: Abstract interpretation
    - name: Local and global patterns
    - name: Partial separability
  - name: Coloring
    subsections:
    - name: Graph formulation
    - name: Greedy algorithm
  - name: Second order
    subsections:
    - name: Hessians
    - name: Hessian-vector products
    - name: Pattern detection
    - name: Symmetric coloring
  - name: Demonstration
    subsections:
    - name: Necessary packages
    - name: Test function
    - name: Dense Jacobian
    - name: Preparation
    - name: Performance benefits
    - name: Coloring visualization


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
    /* Adapted from Andreas Kirsch https://github.com/iclr-blogposts/2024/blob/c111fe06039524fcb60a76c1e9bed26667d30fcf/_posts/2024-05-07-dpi-fsvi.md  */
    .box-note {
        font-size: 14px;
        padding: 15px 15px 10px 15px;
        margin: 20px 20px 20px 10px;
        border-left: 7px solid #009E73;
        border-radius: 5px;
    }
    d-article .box-note {
        background-color: #eee;
        border-left-color: #009E73;
    }
    html[data-theme='dark'] d-article .box-note {
        background-color: #333333;
        border-left-color: #009E73;
    }
---

<!-- LaTeX commands -->
<div style="display: none">
    $$
    \newcommand{\colorf}[1]{\textcolor{RoyalBlue}{#1}}
    \newcommand{\colorh}[1]{\textcolor{RedOrange}{#1}}
    \newcommand{\colorg}[1]{\textcolor{PineGreen}{#1}}
    \newcommand{\colorv}[1]{\textcolor{VioletRed}{#1}}
    \def\sR{\mathbb{R}}
    \def\vx{\mathbf{x}}
    \def\vv{\mathbf{v}}
    \def\vb{\mathbf{e}}
    \newcommand{\vvc}[1]{\colorv{\vv_{#1}}}
    \newcommand{\vbc}[1]{\colorv{\vb_{#1}}}
    \newcommand{\dfdx}[2]{\frac{\partial f_{#1}}{\partial x_{#2}}(\vx)}
    \newcommand{\J}[2]{J_{#1}(#2)} 
    \def\Jf{\J{f}{\vx}}
    \def\Jg{\J{g}{\vx}}
    \def\Jh{\J{h}{g(\vx)}}
    \def\Jfc{\colorf{\Jf}}
    \def\Jgc{\colorg{\Jg}}
    \def\Jhc{\colorh{\Jh}}
    \newcommand{\D}[2]{D{#1}(#2)}
    \def\Df{\D{f}{\vx}}
    \def\Dg{\D{g}{\vx}}
    \def\Dh{\D{h}{g(\vx)}}
    \def\Dfc{\colorf{\Df}}
    \def\Dgc{\colorg{\Dg}}
    \def\Dhc{\colorh{\Dh}}
    $$
</div>

First-order optimization is ubiquitous in Machine Learning (ML) but second-order optimization is much less common.
The intuitive reason is that large gradients are cheap, whereas large Hessian matrices are expensive.
Luckily, in numerous applications of ML to science or engineering, **Hessians (and Jacobians) exhibit sparsity**:
most of their coefficients are known to be zero.
Leveraging this sparsity can vastly **accelerate Automatic Differentiation** (AD) for Hessians and Jacobians,
while decreasing its memory requirements.
Yet, while traditional AD is available in many high-level programming languages,
**automatic sparse differentiation (ASD) is not as widely used**.
One reason is that the underlying theory was developed outside of the ML research ecosystem,
by people more familiar with low-level programming languages.

With this blog post, we aim to shed light on the inner workings of ASD,
thus bridging the gap between the ML and AD communities.
We start out with a short introduction to traditional AD,
covering the computation of Jacobians in both forward and reverse mode.
We then dive into the two primary components of ASD:
**sparsity pattern detection** and **matrix coloring**.
Having described the computation of sparse Jacobians,
we move on to sparse Hessians.  
We conclude with a practical demonstration of ASD,
providing performance benchmarks and guidance on when to use ASD over AD.

## Automatic Differentiation

Let us start by covering the fundamentals of traditional AD.

AD makes use of the **compositional structure** of mathematical functions like deep neural networks.
To make things simple, we will mainly look at a differentiable function $f$
composed of two differentiable functions $g: \mathbb{R}^{n} \rightarrow \mathbb{R}^{p}$ and $h: \mathbb{R}^{p} \rightarrow \mathbb{R}^{m}$,
such that $f = h \circ g: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$.
The insights gained from this toy example should translate directly to more deeply composed functions $f = g^{(L)} \circ g^{(L-1)} \circ \cdots \circ g^{(1)}$.
For ease of visualization, we work in small dimension, but the real benefits of ASD only appear as the dimension grows.

### The chain rule

For a function $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ and a point of linearization $\mathbf{x} \in \mathbb{R}^{n}$,
the Jacobian $J_f(\mathbf{x})$ is the $m \times n$ matrix of first-order partial derivatives, such that the $(i,j)$-th entry is

$$ \big( \Jf \big)_{i,j} = \dfdx{i}{j} \in \sR \quad . $$

For a composed function 

$$ \colorf{f} = \colorh{h} \circ \colorg{g}, $$

the **multivariate chain rule** tells us that we obtain the Jacobian of $f$ by **multiplying** the Jacobians of $h$ and $g$:

$$ \Jfc = \Jhc \cdot \Jgc \quad .$$

Figure 1 illustrates this for $n=5$, $m=4$ and $p=3$.
We will keep using these dimensions in following illustrations.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/chainrule_num.svg" class="img-fluid" %}
<div class="caption">
    Figure 1: Visualization of the multivariate chain rule for $f = h \circ g$.
</div>

### AD is matrix-free

We have seen how the chain rule translates the compositional structure of a function into the product structure of its Jacobian.
Thanks to the small dimensions $n$, $m$ and $p$, this approach worked well on our toy example in Figure 1.
In practice however, there is a problem:
**materializing intermediate Jacobian matrices is inefficient and often impossible**, especially with a dense matrix format.
Examples of dense matrix formats include NumPy's `ndarray`, PyTorch's `Tensor`, JAX's `Array` and Julia's `Matrix`.

As a motivating example, let us take a look at a tiny convolutional layer.
We consider a convolutional filter of size $5 \times 5$, a single input channel and a single output channel.
An input of size $28 \times 28 \times 1$ results in a $576 \times 784$ Jacobian, the structure of which is shown in Figure 2.
All the white coefficients are **structural zeros**.

If we materialize the entire Jacobian as a dense matrix:

- we waste time computing coefficients which are mostly zero;
- we waste memory storing those zero coefficients.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/big_conv_jacobian.png" class="img-fluid" %}
<div class="caption">
    Figure 2: Structure of the Jacobian of a tiny convolutional layer.
</div>

In modern neural network architectures, which can contain over one trillion parameters,
computing intermediate Jacobians is not only inefficient: it exceeds available memory.
AD circumvents this limitation using **linear maps**, lazy operators that act exactly like matrices but without materializing them.

The differential $Df: \mathbf{x} \longmapsto Df(\mathbf{x})$ is a linear map which provides the best linear approximation of $f$ around a given point $\vx$.
We can rephrase  the chain rule as a **composition of linear maps** instead of a product of matrices:

$$ \Dfc = \colorf{\D{(h \circ g)}{\vx}} = \Dhc \circ \Dgc .$$

Note that all terms in this formulation of the chain rule are linear maps.
A new visualization for our toy example can be found in Figure 3b.
Our illustrations distinguish between materialized matrices and linear maps by using solid and dashed lines respectively.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/chainrule_num.svg" class="img-fluid" %}
<div class="caption">
    Figure 3a: Chain rule using materialized Jacobians (solid outline).
</div>

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/matrixfree.svg" class="img-fluid" %}
<div class="caption">
    Figure 3b: Chain rule using matrix-free linear maps (dashed outline).
</div>

<aside class="l-body box-note" markdown="1">
We visualize "matrix entries" in linear maps to build intuition.
Even though following illustrations will sometimes put numbers onto these entries,
linear maps are best thought of as black-box functions.
</aside>

### Forward-mode AD

Now that we have translated the compositional structure of our function $f$ into a compositional structure of linear maps, we can evaluate them by propagating **materialized vectors** through them.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode_eval.svg" class="img-fluid" %}
<div class="caption">
    Figure 4: Evaluating linear maps in forward-mode.
</div>

Figure 4 illustrates the propagation of a vector $\mathbf{v}_1 \in \mathbb{R}^n$ from the right-hand side.
Since we propagate in the order of the original function evaluation, this is called **forward-mode AD**.

In the first step, we evaluate $Dg(\mathbf{x})(\mathbf{v}_1)$.
Since this operation by definition corresponds to 

$$ \vvc{2} = \Dgc(\vvc{1}) = \Jgc \cdot \vvc{1} \;\in \sR^p ,$$

it is also commonly called a **Jacobian-vector product** (JVP) or **pushforward**.
The resulting vector $\vv_2$ is then used to compute the subsequent JVP 

$$ \vvc{3} = \Dhc(\vvc{2}) = \Jhc \cdot \vvc{2} \;\in \sR^m ,$$

which in accordance with the chain rule is equivalent to 

$$ \vvc{3} = \Dfc(\vvc{1}) = \Jfc \cdot \vvc{1} ,$$

the JVP of our composed function $f$.

**Note that we did not materialize intermediate Jacobians at any point** – we only propagated vectors through linear maps.

### Reverse-mode AD

We can also propagate vectors through our linear maps from the left-hand side, 
resulting in **reverse-mode AD**, shown in Figure 5.
It is also commonly called a **vector-Jacobian product** (VJP) or **pullback**.
Just like forward-mode, reverse-mode is also matrix-free: **no intermediate Jacobians are materialized at any point**.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/reverse_mode_eval.svg" class="img-fluid" %}
<div class="caption">
    Figure 5: Evaluating linear maps in reverse-mode.
</div>

### From linear maps back to Jacobians

The linear map formulation allows us to avoid intermediate Jacobian matrices in long chains of function compositions.
But can we use this machinery to materialize the **Jacobian** of the composition $f$ itself?

As shown in Figure 6, we can **materialize Jacobians column by column** in forward mode.
Evaluating the linear map $Df(\mathbf{x})$ on the $i$-th standard basis vector materializes the $i$-th column of the Jacobian $J_f(\mathbf{x})$:

$$ \Dfc(\vbc{i}) = \left( \Jfc \right)_\colorv{i,:} $$

Thus, materializing the full $m \times n$ Jacobian requires one JVP with each of the $n$ standard basis vectors of the **input space**.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode.svg" class="img-fluid" %}
<div class="caption">
    Figure 6: Forward-mode AD materializes Jacobians column-by-column.
</div>

As illustrated in Figure 7, we can also **materialize Jacobians row by row** in reverse mode.
Unlike forward mode in Figure 6,
this requires one VJP with each of the $m$ standard basis vectors of the **output space**.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/reverse_mode.svg" class="img-fluid" %}
<div class="caption">
    Figure 7: Reverse-mode AD materializes Jacobians row-by-row.
</div>

<aside class="l-body box-note" markdown="1">
Since neural networks are usually trained using scalar loss functions,
reverse-mode AD only requires the evaluation of a single VJP to compute a gradient.
This makes it the method of choice for machine learners,
who typically refer to reverse-mode AD as *backpropagation*.
</aside>

## Automatic sparse differentiation

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
    Figure 8: A sparse Jacobian and its corresponding sparse linear map.
</div>

When functions have many inputs and many outputs,
a given output does not always depend on every single input.
This endows the corresponding Jacobian with a **sparsity pattern**,
where zero coefficients denote an absence of (first-order) dependency.
The previous case of a convolutional layer is a simple example.
An even simpler example is an activation function applied elementwise,
for which the Jacobian is the identity matrix.

### Leveraging sparsity

For now, we assume that the sparsity pattern of the Jacobian is always the same, regardless of the input, and that we know it ahead of time.
We say that two columns or rows of the Jacobian matrix are orthogonal if, for every index, at most one of them has a nonzero coefficient.

In other words, the vectors representing their sparsity patterns are structurally orthogonal.
The dot product between these vectors is always zero, regardless of their values.

**The core idea of ASD is that we can materialize multiple orthogonal columns (or rows) in a single product evaluation.**
Since linear maps are additive, it always holds that for a set of basis vectors (columns of the identity matrix),

$$ \Dfc(\vbc{i}+\ldots+\vbc{j}) 
= \underbrace{\Dfc(\vbc{i})}_{\left( \Jfc \right)_\colorv{i,:}} 
+ \ldots
+ \underbrace{\Dfc(\vbc{j})}_{\left( \Jfc \right)_\colorv{j,:}} 
. $$

The components of the sum on the right-hand side each correspond to a column of the Jacobian.
If these columns are known to be **orthogonal**,
the sum can be uniquely decomposed into its components, a process known as **decompression**.
Thus, a single JVP is enough to compute the nonzero coefficients of several columns at once.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_ad.svg" class="img-fluid" %}
<div class="caption">
    Figure 9: Materializing multiple orthogonal columns of a Jacobian in forward-mode.
</div>

This specific example using JVPs corresponds to sparse forward-mode AD 
and is visualized in Figure 9, where all orthogonal columns have been colored in matching hues.
By computing a single JVP with the vector $\mathbf{e}_1 + \mathbf{e}_2 + \mathbf{e}_5$, 
we materialize the sum of the first, second and fifth column of our Jacobian.
Then, we assign the values in the resulting vector back to the appropriate Jacobian entries.
This final decompression step is shown in Figure X.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_ad_forward_full.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_ad_forward_decompression.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure X: Materializing a Jacobian with forward-mode ASD: (a) compressed evaluation of orthogonal columns (b) decompression to Jacobian matrix
</div>

The same idea can also be applied to reverse mode AD, as shown in Figure Y.
Instead of leveraging orthogonal columns, we rely on orthogonal rows.
We can then materialize multiple rows in a single VJP.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_ad_reverse_full.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_ad_reverse_decompression.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure Y: Materializing a Jacobian with reverse-mode ASD: (a) compressed evaluation of orthogonal rows (b) decompression to Jacobian matrix
</div>

### Sparsity pattern detection and coloring

Unfortunately, our initial assumption had a major flaw: 
since AD only gives us a composition of linear maps and linear maps are black-box functions,
the structure of the Jacobian is completely unknown.
In other words, **we cannot tell which rows and columns are orthogonal without first materializing a Jacobian matrix.**
But if we fully materialize a Jacobian via traditional AD, ASD isn't needed at all.

The solution to this problem is shown in Figure 10 (a):
in order to find orthogonal columns (or rows), we don't need to materialize the full Jacobian.
Instead, it is enough to **detect the sparsity pattern** of the Jacobian.
This binary-valued pattern contains enough information to deduce orthogonality.
From there, we use a **coloring algorithm** to group mutually orthogonal columns (or rows) together.
Such a coloring can be visualized on Figure 10 (b), 
where the yellow columns will be evaluated together (first JVP) 
and the light blue ones will be evaluated together (second JVP), 
for a total of 2 JVPs instead of 5.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparsity_pattern.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/coloring.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 10: The first two steps of ASD: (a) sparsity pattern detection, (b) coloring of the sparsity pattern.
</div>

To sum up, ASD consists of four steps:

1. Pattern detection
2. Coloring
3. Compressed AD
4. Decompression

We now describe the first two steps in more detail.
Usually, these steps are much slower than a single call to the function $f$, but much faster than a full computation of the Jacobian with AD.
This makes the sparse procedure worth it even for moderately large matrices.
Additionally, if we need to compute Jacobians multiple times (for different inputs) and are able to reuse the sparsity pattern and the coloring result, 
the cost of this prelude can be amortized over time.

## Pattern detection

Sparsity pattern detection can be thought of as a binary version of AD.
Mirroring the diversity of existing approaches to AD,
there are also many possible approaches to sparsity pattern detection,
each with their own advantages and tradeoffs.

The method we will present here corresponds to a binary forward-mode AD system. 
in which performance is gained by representing matrix rows as index sets.

<aside class="l-body box-note" markdown="1">
<!-- TODO: cite a wide list of approaches here -->
Alternatives include Bayesian probing, ... *TODO* 
</aside>

### Index sets

Our goal with sparsity pattern detection is to quickly materialize the binary pattern of the Jacobian.
One way to achieve better performance than traditional AD is to encode row sparsity patterns as index sets.
The $i$-th row of the Jacobian is given by 

$$ \big( \Jf \big)_{i,:} 
= \left[\dfdx{i}{j}\right]_{1 \le j \le n}
= \begin{bmatrix}
    \dfdx{i}{1} &
    \ldots      &
    \dfdx{i}{n}
\end{bmatrix} .
$$

However, since we are only interested in the binary pattern 

$$ \left[\dfdx{i}{j} \neq 0\right]_{1 \le j \le n} , $$

we can instead represent the sparsity pattern of the $i$-th column of a Jacobian by the corresponding **index set of non-zero values**

$$ \left\{j \;\Bigg|\; \dfdx{i}{j} \neq 0\right\} . $$

These equivalent sparsity pattern representations are illustrated in Figure 11.
Each row index set tells us **which inputs influenced a given output**, at the first-order. For instance, output $i=2$ was influenced by inputs $j=4$ and $j=5$.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparsity_pattern_representations.svg" class="img-fluid" %}
<div class="caption">
    Figure 11: Sparsity pattern representations: (a) original matrix, (b) binary pattern, (c) row index sets.
</div>

### Efficient propagation

Figure 12 shows the traditional forward-AD pass we want to avoid:
propagating a full identity matrix through a linear map would materialize the Jacobian of $f$, 
but also all intermediate linear maps.
As previously discussed, this is not a viable option due to its inefficiency and high memory requirements.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode_naive.svg" class="img-fluid" %}
<div class="caption">
    Figure 12: Materializing a Jacobian forward-mode. 
    Due to high memory requirements for intermediate Jacobians, this approach is inefficient or impossible.  
</div>

Instead, we initialize an input vector with index sets corresponding to the identity matrix. 
An alternative view on this vector is that it corresponds to the index set representation of the Jacobian of the input, since $\frac{\partial x_i}{\partial x_j} \neq 0$ only holds for $i=j$.

Our goal is to propagate this index set such that we get an output vector of index sets 
that corresponds to the Jacobian sparsity pattern.
This idea is visualized in Figure 13.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode_sparse.svg" class="img-fluid" %}
<div class="caption">
    Figure 13: Propagating an index set through a linear map to obtain a sparsity pattern.  
</div>

### Abstract interpretation

Instead of going into implementation details,
we want to provide some intuition on the second key ingredient of our forward-mode sparsity detection system: 
**abstract interpretation**.

We will demonstrate this on a second toy example, the function

$$ f(\vx) = \begin{bmatrix}
x_1 x_2 + \text{sgn}(x_3)\\
\text{sgn}(x_3) \frac{x_4}{2}
\end{bmatrix} \, .$$

The corresponding computational graph is shown in Figure 14,
where circular nodes correspond to elementary operators,
in this case addition, multiplication, division and the sign function.
Scalar inputs $x_i$ and outputs $y_j$ are shown in rectangular nodes.
Instead of evaluating the original compute graph for a given input $\mathbf{x}$,
<!-- (also called *primal computation*) -->
all inputs are seeded with their respective input index sets.
Figure 14 annotates these index sets on the edges of the computational graph.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/compute_graph.png" class="img-fluid" %}
<div class="caption">
    Figure 14: Computational graph of the function $ f(\vx) = x_1 + x_2x_3 + \text{sgn}(x_4) $, annotated with corresponding index sets.  
</div>

Our sparsity detection system must now perform **abstract interpretation** of our computational graph.
Instead of computing the original function, 
each operator must correctly propagate and accumulate the index sets of its inputs, 
depending on whether an operator globally has a non-zero derivative or not.  

Since addition, multiplication and division globally have non-zero derivatives with respect to both of their inputs,
the index sets of their inputs are accumulated and propagated. 
The sign function has a zero-valued derivative for any input value. 
It therefore doesn't propagate the index set of its input. 
Instead, it returns an empty set.

Figure 14 shows the resulting output index sets $\\{1, 2\\}$ and $\\{4\\}$ for outputs 1 and 2 respectively.
These match the analytic Jacobian

$$ J_f(x) = \begin{bmatrix}
x_2 & x_1 & 0 & 0\\
  0 &   0 & 0 & \frac{\text{sgn}(x_3)}{2}
\end{bmatrix} \, .
$$

### Local and global patterns

The type of abstract interpretation shown above corresponds to *global sparsity detection*,
computing index sets 

$$ \left\{j \;\Bigg|\; \dfdx{i}{j} \neq 0,\, x \in \sR^{n} \right\} $$

over the entire input domain.
Another type of abstract interpretation can be implemented, 
in which the original *primal computation* is propagated alongside index sets, computing 

$$ \left\{j \;\Bigg|\; \dfdx{i}{j} \neq 0 \right\} $$

for a given input $\mathbf{x}$. 
These *local sparsity patterns* are strict subsets of global sparsity patterns,
and can therefore result in fewer colors.
However, they need to be recomputed when changing the input.

### Partial separability

When we know in advance that the function has partial separability,
sparsity pattern detection becomes significantly more efficient.
Partial separability means that the function can be decomposed into independent or weakly dependent subcomponents, often corresponding to blocks in the Jacobian matrix.
This structure allows the sparsity pattern to be identified separately for each block, rather than considering the full Jacobian matrix as a whole.

## Coloring

Once we have detected a sparsity pattern, our next goal is to figure out how to group the columns (or rows) of the Jacobian.
The columns (or rows) in each group will be evaluated simultaneously with a single JVP (or VJP), where the vector is a linear combination of basis vectors called a **seed**.
If they are mutually orthogonal, then this gives all the necessary information to retrieve every nonzero coefficient of the matrix.

### Graph formulation

Luckily, this can be reformulated as a graph coloring problem, which is very well studied.
Let us build a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with vertex set $\mathcal{V}$ and edge set $\mathcal{E}$, 
such that each column is a vertex of the graph, and two vertices are connected if and only if their respective columns share a non-zero index.
Put differently, an edge between vertices $j_1$ and $j_2$ means that columns $j_1$ and $j_2$ are not orthogonal.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/colored_graph.svg" class="img-fluid" %}
<div class="caption">
    Figure X: Optimal graph coloring.
</div>

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/colored_graph_suboptimal.svg" class="img-fluid" %}
<div class="caption">
    Figure X: Suboptimal graph coloring. Node 1 could be colored in yellow, leading to redundant computations of matrix-vector products.
</div>

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/colored_graph_infeasible.svg" class="img-fluid" %}
<div class="caption">
    Figure X: Infeasible graph coloring. Nodes 2 and 4 on the graph are adjacent, but share a color. This results in overlapping columns.
</div>

We want to assign to each vertex $j$ a color $c(j)$, such that any two adjacent vertices $(j_1, j_2) \in \mathcal{E}$ have different colors $c(j_1) \neq c(j_2)$.
This constraint ensures that columns in the same color group are indeed orthogonal.
If we can find a coloring which uses the smallest possible number of distinct colors, it will minimize the number of groups, and thus the computational cost of the AD step.

If we perform column coloring, forward-mode AD is required, while reverse-mode AD is needed for row coloring.
Note that more advanced coloring techniques could use both modes, such as **bicoloring**.

<aside class="l-body box-note" markdown="1">
<!-- TODO -->
There are more efficient representations, e.g. *TODO*
</aside>

### Greedy algorithm

Unfortunately, the graph coloring problem is NP-hard, meaning that there is (probably) no way to solve it polynomially for every instance.
The optimal solution is known only for specific patterns, such as banded matrices.
However, efficient heuristics exist that generate good enough solutions in reasonable time.
The most widely used heuristic is the greedy algorithm, which processes vertices one after the other.
This algorithm assigns to each vertex the smallest color that is not already present among its neighbors, and it never backtracks.
A crucial hyperparameter is the choice of ordering, for which various criteria have been proposed.

## Second order

While first-order automatic differentiation AD focuses on computing the gradient or Jacobian, second-order AD extends this by involving the **Hessian**.

### Hessians

The **Hessian** contains second-order partial derivatives of a scalar function, essentially capturing the curvature of the function at a point.
This is particularly relevant in **optimization**, where the Hessian provides crucial information about the nature of the function's local behavior.
Specifically, the Hessian allows us to distinguish between local minima, maxima, and saddle points.
By incorporating second-order information, optimization algorithms converge more robustly in cases where the gradient alone doesn't provide enough information for effective search directions.
This is especially useful in **nonlinear optimization problems**.

### Hessian-vector products

In the context of automatic differentiation, the key operation is **Hessian-vector product (HVP)**.
The Hessian $\nabla^2 f(\mathbf{x})$ is the Jacobian matrix of the gradient $\nabla f$:

$$ \nabla^2 f (\mathbf{x}) = J_{\nabla f}(\mathbf{x}) $$

An HVP computes the product of the Hessian matrix with a vector, which can be viewed as the JVP of the gradient (a gradient which is itself computed via a VJP of $f$):

$$ \nabla^2 f(x) (\mathbf{v}) = D[\nabla f](\mathbf{x})(\mathbf{v}) $$

Thus it is quite common to say that HVPs are computed with "forward over reverse" AD.
The complexity of a single HVP scales roughly with the complexity of the function $f$ itself.

The Hessian has a **symmetric** structure (equal to its transpose), which means that matrix-vector products and vector-matrix products coincide.
This specificity can be exploited in the sparsity detection as well as in the coloring phase.

### Pattern detection

Detecting the sparsity pattern of the Hessian is more complicated than for the Jacobian.
This is because, in addition to the usual linear dependencies, we now have to account for **nonlinear interactions** between every pair of coefficients.
For instance, if $f(x)$ involves a term of the form $x_1 + x_2$, it will not affect the Hessian. 
On the other hand, a term $x_1 x_2$ will yield two equal non-zero coefficients, one at position $(1, 2)$ and one at position $(2, 1)$.
Thus, the abstract interpretation system used for detection needs a finer classification of operators, to distinguish between locally linear ones (sum, max) and locally nonlinear ones (product, exp).

### Symmetric coloring

When it comes to **graph coloring** for the Hessian, the process can be more efficient than those for the Jacobian due to its **symmetry**.
Even if two columns in the Hessian are not orthogonal, missing coefficients can be recovered by leveraging the corresponding rows instead of relying solely on the columns.
In other words, if $H_{ij}$ is lost during compression because of colliding nonzero coefficients, there is still a chance to retrieve it through $H_{ji}$.
This symmetry enables **colorings with fewer colors**, reducing the complexity of the AD part compared to traditional row or column coloring.

While the **decompression** step for symmetric coloring is more computationally expensive, this cost is typically negligible compared to the overhead of computing HVPs.
Moreover, symmetric coloring becomes especially advantageous when the Hessian needs to be recomputed for multiple values of $x$, as the reduced number of colors amortizes the initial expense.

## Demonstration

We conclude this blog post with a demonstration of automatic sparse differentiation in a high-level programming language, namely the [Julia language](https://julialang.org/).
While still at an early stage of development, we hope that such an example of unified pipeline for sparse Jacobians and Hessians can inspire developers in other languages to revisit ASD.

<aside class="l-body box-note" markdown="1">
The authors of this blog post are all developers of the ASD ecosystem in Julia. We are not aware of a similar ecosystem in Python or R, which is why we chose Julia to present it.
</aside>

### Necessary packages

Here are the packages we will use for this demonstration.
We also use a few other packages for visualization.

| Package | Purpose |
|---|---|
| [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl) | Sparsity pattern detection with operator overloading. |
| [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl) | Greedy algorithms for colorings, decompression utilities. |
| [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) | Forward-mode AD and computation of JVPs. |
| [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) | High-level interface bringing all of these together. |

As with any language, the first step is importing the dependencies.

```julia
using DifferentiationInterface
using SparseConnectivityTracer, SparseMatrixColorings
import ForwardDiff
```

### Test function

As our test function, we choose a very simple iterated difference operator.
It takes a vector $\mathbf{x} \in \mathbb{R}^n$ and outputs a slightly shorter vector $y \in \mathbb{R}^{n-k}$ depending on the number of iterations $k$.
In pure Julia, this is written as follows (using the built-in `diff` recursively):

```julia
iter_diff(x, k) = k == 0 ? x : diff(iter_diff(x, k-1))
```

Let us check that the function returns what we expect:

```julia
julia> iter_diff([1, 4, 9, 16], 1)
3-element Vector{Int64}:
 3
 5
 7

julia> iter_diff([1, 4, 9, 16], 2)
2-element Vector{Int64}:
 2
 2
```

### Dense Jacobian

The key concept behind DifferentiationInterface.jl is that of *backends*.
There are several AD systems in Julia, each with different features and tradeoff, that can be accessed them through a common API.
Here, we use ForwardDiff.jl as our AD backend:

```julia
dense_backend = AutoForwardDiff()
```

To build a sparse backend, we bring together three ingredients corresponding to the various phases of ASD:

```julia
sparsity_detector = TracerSparsityDetector()  # from SparseConnectivityTracer
coloring_algorithm = GreedyColoringAlgorithm()  # from SparseMatrixColorings
sparse_backend = AutoSparse(dense_backend; sparsity_detector, coloring_algorithm)
```

We can now obtain the Jacobian of `iter_diff` (with respect to $\mathbf{x}$) using either backend, and compare the results:

```julia
julia> x, k = rand(10), 3;

julia> jacobian(iter_diff, dense_backend, x, Constant(k))
7×10 Matrix{Float64}:
 -1.0   3.0  -3.0   1.0   0.0   0.0   0.0   0.0   0.0  0.0
  0.0  -1.0   3.0  -3.0   1.0   0.0   0.0   0.0   0.0  0.0
  0.0   0.0  -1.0   3.0  -3.0   1.0   0.0   0.0   0.0  0.0
  0.0   0.0   0.0  -1.0   3.0  -3.0   1.0   0.0   0.0  0.0
  0.0   0.0   0.0   0.0  -1.0   3.0  -3.0   1.0   0.0  0.0
  0.0   0.0   0.0   0.0   0.0  -1.0   3.0  -3.0   1.0  0.0
  0.0   0.0   0.0   0.0   0.0   0.0  -1.0   3.0  -3.0  1.0

julia> jacobian(iter_diff, sparse_backend, x, Constant(k))
7×10 SparseArrays.SparseMatrixCSC{Float64, Int64} with 28 stored entries:
 -1.0   3.0  -3.0   1.0    ⋅     ⋅     ⋅     ⋅     ⋅    ⋅ 
   ⋅   -1.0   3.0  -3.0   1.0    ⋅     ⋅     ⋅     ⋅    ⋅ 
   ⋅     ⋅   -1.0   3.0  -3.0   1.0    ⋅     ⋅     ⋅    ⋅ 
   ⋅     ⋅     ⋅   -1.0   3.0  -3.0   1.0    ⋅     ⋅    ⋅ 
   ⋅     ⋅     ⋅     ⋅   -1.0   3.0  -3.0   1.0    ⋅    ⋅ 
   ⋅     ⋅     ⋅     ⋅     ⋅   -1.0   3.0  -3.0   1.0   ⋅ 
   ⋅     ⋅     ⋅     ⋅     ⋅     ⋅   -1.0   3.0  -3.0  1.0
```

In one case, we get a dense matrix, in the other it is sparse.
Note that in Julia, linear algebra operations are optimized for sparse matrices, which means this format can be beneficial for downstream use.
We now show that sparsity also unlocks faster computation of the Jacobian itself.

### Preparation

Sparsity pattern detection and matrix coloring are performed in a so-called "preparation step", whose output can be reused across several calls to `jacobian` (as long as the pattern stays the same).

Thus, to extract more performance, we can create this object once

```julia
prep = prepare_jacobian(iter_diff, sparse_backend, x, Constant(k));
```

and then reuse it as much as possible, for instance inside the loop of an iterative algorithm (note the additional `prep` argument):

```julia
jacobian(iter_diff, prep, sparse_backend, x, Constant(k))
```

Inside the preparation result, we find the result of sparsity pattern detection

```julia
julia> sparsity_pattern(prep)
7×10 SparseArrays.SparseMatrixCSC{Bool, Int64} with 28 stored entries:
 1  1  1  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  1  1  1  1  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  1  1  1  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  1  1  1  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  1  1  1  1  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  1  1  1  1  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  1  1  1
```

and the coloring of the columns:

```julia
julia> column_colors(prep)
10-element Vector{Int64}:
 1
 2
 3
 4
 1
 2
 3
 4
 1
 2
```

Note that it uses only $c = 4$ different colors, which means we need $4$ JVPs instead of the initial $n = 10$ to reconstruct the Jacobian.

```julia
julia> ncolors(prep)
4
```

This discrepancy typically gets larger as the input grows: it is not rare for the number of columns to be a constant that does not depend on $n$.
It is the key driver of ASD performance.

### Performance benefits

Here we present a benchmark for a slightly larger input, $n = 1000$ and $k = 10$.
It can be obtained with the following code:

```julia
using DifferentiationInterfaceTest
scen = Scenario{:jacobian,:out}(iter_diff, rand(1000); contexts=(Constant(10),))
data = benchmark_differentiation([dense_backend, sparse_backend], [scen]; benchmark=:full)
```

In the table below:

- the column "sparse" tells us which backend we are using
- the column "prepared" tells us whether or not preparation is included in the measurements (for dense AD preparation is essentially trivial)
- the column "time" contains the execution time in seconds
- the column "bytes" contains the allocated memory in bytes

| **sparse** | **prepared** | **time**  | **bytes** |
|-----------:|-------------:|----------:|----------:|
| false      | true         | 2.732e-02 | 1.679e+08 |
| false      | false        | 2.183e-02 | 1.679e+08 |
| true       | true         | 1.923e-04 | 1.943e+06 |
| true       | false        | 2.995e-03 | 1.323e+07 |

<!-- TODO: update benchmarks to new function (re-run Pluto) -->

As shown in the table, even when we include the overhead of pattern detection and coloring, the sparse backend is around $5 \times$ faster than the dense backend.
The speedup becomes $100 \times$ once we discard this overhead, which can be amortized over several `jacobian` computations.

### Coloring visualization

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/demo/banded.png" class="img-fluid" %}

<!-- TODO: add comments -->