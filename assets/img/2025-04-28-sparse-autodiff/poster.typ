#import "@preview/peace-of-posters:0.5.1": *

#let theme-bifold = (
  "heading-box-args": (
    inset: 0.6em,
    width: 100%,
    fill: rgb("#a3dce8"),
    stroke: rgb("#a3dce8"),
  ),
  "heading-text-args": (
    fill: rgb("#002f67"),
  ),
  "body-box-args": (
    inset: 0.6em,
    width: 100%,
    fill: rgb("#ffffff"),
    stroke: rgb("#ffffff"),
  ),
  "body-text-args": (:),
)

#let bifold-a0-poster(doc) = [
  #set page("a0", margin: 3cm, background: image("bifold/BIFOLD_bg.png"))
  #set text(
    font: ("Catamaran", "Helvetica", "Arial"),
    size: layout-a0.at("body-size"),
  )
  #let box-spacing = 1.5cm
  #set columns(gutter: box-spacing)
  #set block(spacing: box-spacing)
  #set-poster-layout(layout-a0)
  #set-theme(theme-bifold)
  #update-poster-layout(
    spacing: box-spacing,
    heading-size: 40pt,
    title-size: 120pt,
  )
  #doc
]

// By default, show images centered and at 70% width
#let my-image(source, width: 70%) = {
  align(center, image(source, width: width))
}

// Use BIFOLD template we just defined
#show: bifold-a0-poster

#title-box(
  [
    // #image("bifold/footer-bg.png", width: 100%)
    Leverging Sparsity\ for Fast Jacobians and Hessians
  ],
  subtitle: [An Illustrated Guide to Automatic Sparse Differentiation],
  authors: [
    Adrian Hill#super("1,2"),
    Guillaume Dalle#super("3"),
    Alexis Montoison#super("4")
  ],
  institutes: [
    #super("1")BIFOLD – Berlin Institute for the Foundations of Learning and Data, Berlin, Germany,
    #super("2")Machine Learning Group, Technical University of Berlin, Berlin, Germany,
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
      #my-image("chainrule_num.svg")
      #my-image("matrixfree.svg")
    ]

    #column-box(heading: [Forward and reverse mode])[
      Foobar
      #my-image("forward_mode_eval.svg")
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

    #column-box(heading: [Pattern detection])[
      Foobar
      #my-image("sparsity_pattern.svg", width: 40%)
    ]


    #column-box(heading: [Coloring])[
      Foobar
      #my-image("colored_graph.svg")
    ]

    #column-box(heading: [Bicoloring])[
      Foobar
      #my-image("bicoloring.svg", width: 40%)
    ]



    #column-box(
      heading: [Demonstration],
      stretch-to-next: true,
    )[
      #show raw: it => block(
        fill: rgb("#ffffff"),
        inset: 8pt,
        radius: 5pt,
        text(fill: rgb("#111111"), it),
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
  Bottom text goes here.
]
