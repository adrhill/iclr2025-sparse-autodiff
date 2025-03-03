#import "@preview/peace-of-posters:0.5.1": *

// Next, we specify some general settings formatting settings.
// #set page("a1", margin: 2cm)
// #set-poster-layout(layout-a1)
#set page("a0", margin: 1cm)
#set-poster-layout(layout-a0)
#set text(font: "Arial", size: layout-a0.at("body-size"))

#let box-spacing = 1.2em
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#update-poster-layout(spacing: box-spacing)// , heading-size: 10pt)

// After that we choose a predefined theme.
#set-theme(uni-fr)

#title-box(
  [Leverging Sparsity for Fast Jacobians and Hessians],
  subtitle: [An Illustrated Guide to Automatic Sparse Differentiation],
  authors: [
    Adrian Hill#super("1,2"),
    Guillaume Dalle#super("3"),
    Alexis Montoison#super("4")
  ],
  institutes: [
    #super("1")BIFOLD – Berlin Institute for the Foundations of Learning and Data, Berlin, Germany,
    #super("2")
    Machine Learning Group, Technical University of Berlin, Berlin, Germany,
    #super("3")LVMT, ENPC, Institut Polytechnique de Paris, Univ Gustave Eiffel, Marne-la-Vallée, France
    #super("4")Argonne National Laboratory
  ],
  // keywords: [Automatic Differentiation, Sparsity, Second-order optimization],
)

#columns(
  2,
  [
    // First column
    #column-box(heading: [Matrices vs. Operators])[
      Foobar
      #image("chainrule_num.svg")
      #image("matrixfree.svg")
    ]

    #column-box(heading: [Forward and reverse mode])[
      Foobar
      #image("forward_mode_eval.svg")
      #image("forward_mode.svg")
    ]


    #column-box(
      heading: [Stretching],
      stretch-to-next: true,
    )[
      Foobar
    ]

    #colbreak()

    // Second column
    #column-box(heading: [Automatic Sparse Differentiation])[
      Foobar
      #image("sparse_ad_forward_full.svg")
      #image("colored_graph.svg")
    ]

    #column-box(heading: [Pattern detection])[
      Foobar
      #image("sparsity_pattern.svg")
    ]


    #column-box(heading: [Coloring])[
      Foobar
      #image("colored_graph.svg")
    ]

    #column-box(heading: [Bicoloring])[
      Foobar
      #image("bicoloring.svg")
    ]



    #column-box(
      heading: [Demonstration],
      stretch-to-next: true,
    )[
      #show raw: it => block(
        fill: rgb("#1d2433"),
        inset: 8pt,
        radius: 5pt,
        text(fill: rgb("#a2aabc"), it),
      )
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

      jacobian(f, ad_backend,  x)
      jacobian(f, asd_backend, x)

      ```
    ]
  ],
)


#bottom-box()[
  Align them to the bottom.
]
