#import "bifold.typ": *

// By default, show images centered and at 70% width
#let my-image(source, width: 70%) = {
  align(center, image(source, width: width))
}

// Use BIFOLD template we just defined
#show: bifold-a0-poster

#bifold-title-box(
  [
    // #image("bifold/footer-bg.png", width: 100%)
    Faster Jacobians and Hessians\ by Leveraging Sparsity
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
  // logo: image("bifold/BIFOLD_Logo_negativ_farbig.svg"),
)

#columns(
  2,
  [
    // First column
    #column-box(heading: [Matrices vs. Operators])[
      Foobar
      #my-image("chainrule_num.svg")
      #my-image("matrixfree.svg")
    ]

    #column-box(heading: [Forward and Reverse Mode])[
      Foobar
      #my-image("forward_mode_eval.svg", width: 50%)
      #my-image("forward_mode.svg")
    ]


    #column-box(
      heading: [Automatic Sparse Differentiation],
      stretch-to-next: true,
    )[
      Foobar
      #my-image("sparse_ad_forward_full.svg", width: 55%)
    ]

    #colbreak()

    // Second column

    #column-box(heading: [Pattern Detection])[
      Foobar
      #my-image("sparsity_pattern.svg", width: 35%)
    ]


    #column-box(heading: [Coloring])[
      Foobar
      #my-image("colored_graph.svg")
    ]

    #column-box(heading: [Bicoloring])[
      Foobar
      #my-image("bicoloring.svg", width: 35%)
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
        using SparseConnectivityTracer: TracerSparsityDetector
        using SparseMatrixColorings: GreedyColoringAlgorithm
        import ForwardDiff

        ad_backend = AutoForwardDiff()

        asd_backend = AutoSparse(ad;
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


#bottom-box()[
  Bottom text goes here.
]
