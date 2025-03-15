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
    #super("1")BIFOLD – Berlin Institute for the Foundations of Learning and Data, Berlin, Germany,
    #super("2")Machine Learning Group, Technical University of Berlin, Berlin, Germany,\
    #super("3")LVMT, ENPC, Institut Polytechnique de Paris, Univ Gustave Eiffel, Marne-la-Vallée, France,
    #super("4")Argonne National Laboratory
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
      The chain rule tells us that
      the Jacobian of a composed function $f = h compose g$ is obtained by multiplying the *Jacobian matrices* (solid) of $h$ and $g$.
      #my-image("chainrule_num.svg")

      However, AD doesn't use Jacobian matrices, instead opting for matrix-free *Jacobian operators* (dashed). The chain rule now corresponds to a composition of operators.
      #my-image("matrixfree.svg")

      To turn such (composed) *Jacobian operators* into *Jacobian matrices*,
      they are evaluated with all standard basis vectors.
      #my-image("forward_mode.svg", width: 90%)

      This either constructs matrices column-by-column
      #text("(forward mode, computing as many JVPs as there are inputs)", fill:bifold-gray-2) or row-by-row
      #text(
        "(reverse mode, computing as many VJPs as there are outputs)",
        fill: bifold-gray-2,
      )
      .
      // The computational cost therefore depends on the input and output dimensionality of $f$.
    ]


    #column-box(
      heading: [#capsify("Idea:") Automatic Sparse Differentiation (ASD)],
      stretch-to-next: true,
    )[
      Since Jacobian operators are linear maps,
      we can:
      1. simultaneously compute the values of orthogonal columns/rows
      2. decompress the resulting vectors into the Jacobian matrix.

      #grid(
        columns: 2,
        align: horizon + center,
        column-gutter: 1em,
        image("sparse_ad_forward_full.svg", width: 100%),
        image("sparse_ad_forward_decompression.svg", width: 100%),
      )

      Unfortunately, contrary to our illustrations,
      Jacobian operators (dashed) are black-box functions with unknown structure.
      Two preliminary steps are therefore required to determine orthogonal columns/rows.
    ]

    #colbreak()

    // Second column

    #column-box(heading: [#capsify("Step 1:") Pattern Detection])[
      To find orthogonal colomns, the sparsity pattern of non-zero values in the Jacobian matrix has to be detected.
      This requires a fast binary AD system.
      #my-image("sparsity_pattern.svg", width: 30%)
    ]


    #column-box(heading: [#capsify("Step 2:") Coloring])[
      Graph coloring algorithms are applied to the sparsity pattern to detect orthogonal columns/rows.
      #my-image("colored_graph.svg")
    ]

    #column-box(heading: [Bicoloring])[
      ASD can be accelerated even further
      by coloring both rows and columns
      and combining forward and reverse modes.
      #my-image("bicoloring.svg", width: 30%)
    ]



    #column-box(heading: [Demonstration])[
      #show raw: it => block(
        fill: rgb("#ffffff"),
        inset: 8pt,
        radius: 5pt,
        text(fill: rgb("#111111"), it),
      )
      #pad(
        left: 1em,
        ```julia
        using DifferentiationInterface
        using SparseConnectivityTracer, SparseMatrixColorings
        import ForwardDiff

        ad_backend = AutoForwardDiff()
        asd_backend = AutoSparse(
            ad_backend;
            TracerSparsityDetector(),
            GreedyColoringAlgorithm()
        )

        jacobian(f, ad_backend,  x) # dense
        jacobian(f, asd_backend, x) # sparse

        ```,
      )]

    #column-box(
      heading: [References],
      stretch-to-next: true,
    )[
      Bibliography goes here
    ]
  ],
)


#bottom-box(
  logo: grid(
    columns: 5,
    align: horizon + center,
    column-gutter: 1em,
    image("logos/BIFOLD_Logo_farbig.svg", height: 1.5em),
    image("logos/TUB-color.svg", height: 1.5em),
    image("logos/logo-enpc-ip-rvb.png", height: 2.0em),
    image("logos/LVMT LOGO.png", height: 1.2em),
    image("logos/Argonnelablogo.png", height: 1.5em),
  ),
)[
  Check out our ICLR blog post\
  for more information!
]
