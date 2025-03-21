#import "bifold.typ": *

// By default, show images centered and at 70% width
#let my-image(source, width: 70%) = {
  align(horizon + center, image(source, width: width))
}

// Use BIFOLD template we just defined
#show: bifold-a0-poster

#bifold-title-box(
  [
    Fast Jacobians and Hessians\ by Leveraging Sparsity
  ],
  subtitle: [An Illustrated Guide to Automatic Sparse Differentiation],
  authors: [
    Adrian Hill#super("1,2"),
    Guillaume Dalle#super("3") and
    Alexis Montoison#super("4")
  ],
  institutes: [
    #super("1")BIFOLD – Berlin Institute for the Foundations of Learning and Data, Berlin, Germany,\
    #super("2")Machine Learning Group, Technical University of Berlin, Berlin, Germany,\
    #super("3")LVMT, ENPC, Institut Polytechnique de Paris, Univ Gustave Eiffel, Marne-la-Vallée, France,\
    #super("4")Argonne National Laboratory, Lemont, USA
  ],
  // keywords: [Automatic Differentiation, Sparsity, Second-order optimization],
  authors-size: authors-size,
  institutes-size: institutes-size,
)

#let capsify(body) = {
  text(smallcaps(body), fill: black)
}

#columns(
  2,
  [
    // First column
    #column-box(heading: [#capsify("Recap:") Automatic Differentiation (AD)])[
      The use of AD in deep learning is ubiquitous:
      Instead of having to compute gradients and Jacobians by hand, AD automatically computes them given PyTorch, JAX or Julia code.

      *Matrix-free Jacobian operators* (dashed) lie at the core of AD.
      While we illustrate them as matrices to provide intuition, they are best thought of as *black-box functions* with unknown structure.\
      To turn such Jacobian operators into *Jacobian matrices* (solid),
      they are evaluated with all standard basis vectors.
      #my-image("forward_mode.svg", width: 90%)

      This constructs Jacobian matrices column-by-column#super("1") or row-by-row#super("2").

      #text(
        [#super("1") Forward mode, computing as many JVPs as there are inputs (pictured).],
        fill: bifold-gray-2,
      )\
      #text(
        [#super("2") Reverse mode, computing as many VJPs as there are outputs.],
        fill: bifold-gray-2,
      )
      // The computational cost therefore depends on the input and output dimensionality of $f$.
    ]


    #column-box(
      heading: [Automatic Sparse Differentiation (ASD)],
    )[
      Since Jacobian operators are linear maps,
      we can *simultaneously compute the values of multiple orthogonal columns* (or rows)
      and decompress the resulting vectors into the Jacobian matrix @griewankEvaluatingDerivativesPrinciples2008 @gebremedhinWhatColorYour2005.

      #grid(
        columns: 2,
        align: horizon + center,
        column-gutter: 1em,
        image("sparse_ad_forward_full.svg", width: 100%),
        image("sparse_ad_forward_decompression.svg", width: 100%),
      )

      *To do this, ASD requires knowledge of the structure of the resulting Jacobian matrix.*
      Since Jacobian operators have unknown structure,
      two preliminary steps are required.
    ]

    #bibliography-box(
      "2025-04-28-sparse-autodiff.bib",
      body-size: 13.3pt,
      stretch-to-next: true,
    ) // peace-of-poster seems to have a bug that requires sticking the bibfile into it's source folder.

    #colbreak()

    // Second column

    #column-box(heading: [#capsify("Step 1:") Sparsity Pattern Detection])[
      To find orthogonal columns, the pattern of non-zero values in the Jacobian matrix has to be computed.
      This requires a binary AD system.
      #my-image("sparsity_pattern.svg", width: 28%)

      Mirroring the multitude of approaches to AD,
      many viable approaches to pattern detection exist
      @dixonAutomaticDifferentiationLarge1990
      @bischofEfficientComputationGradients1996
      @waltherComputingSparseHessians2008.
    ]

    #column-box(heading: [#capsify("Step 2:") Coloring])[
      Graph coloring algorithms are applied to the sparsity pattern to group together orthogonal columns/rows @gebremedhinWhatColorYour2005.
      #my-image("colored_graph.svg")
    ]

    #column-box(heading: [Bicoloring])[
      ASD can be accelerated even further
      by coloring both rows and columns
      and combining forward and reverse modes
      @hossainComputingSparseJacobian1998
      @colemanEfficientComputationSparse1998.
      #my-image("bicoloring.svg", width: 28%)
    ]

    #column-box(heading: [Benchmarks], stretch-to-next: true)[
      ASD can drastically outperform AD.
      The performance depends on the sparsity of the Jacobian matrix:
      the cost of sparsity pattern detection and coloring has to be amortized by having to compute fewer matrix-vector products.
      #my-image("demo/benchmark.png", width: 53%)
    ]
  ],
)

// #bottom-box(
//   logo: grid(
//     columns: 5,
//     align: horizon + center,
//     column-gutter: 1em,
//     image("logos/BIFOLD_Logo_farbig.svg", height: 1.5em),
//     image("logos/TUB-color.svg", height: 1.5em),
//     image("logos/logo-enpc-ip-rvb.png", height: 2.0em),
//     image("logos/LVMT LOGO.png", height: 1.2em),
//     image("logos/Argonnelablogo.png", height: 1.5em),
//   ),
// )[
//   Check out our ICLR blog post\
//   for more information & code!
// ]

#bottom-box()[
  #set align(center)
  #grid(
    columns: 5,
    align: horizon + center,
    column-gutter: 1em,
    image("logos/BIFOLD_Logo_farbig.svg", height: 1.65em),
    image("logos/TUB-color.svg", height: 1.65em),
    image("logos/logo-enpc-ip-rvb.png", height: 2em),
    image("logos/LVMT LOGO.png", height: 1.5em),
    image("logos/Argonnelablogo.png", height: 1.75em),
  )
]
