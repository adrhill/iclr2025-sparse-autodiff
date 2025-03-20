#import "@preview/peace-of-posters:0.5.3": *

// BIFOLD CI colors
#let bifold-blue-1 = rgb("#002f67")
#let bifold-blue-2 = rgb("#1abfd5")
#let bifold-blue-3 = rgb("#6fcdde")
#let bifold-blue-4 = rgb("#a3dce8")
#let bifold-blue-5 = rgb("#a3dce8")
#let bifold-blue-6 = rgb(214, 234, 242)


#let bifold-accent-1 = rgb("#bfd330")
#let bifold-accent-2 = rgb("#c40d1e")
#let bifold-accent-3 = rgb("#f4b11c")

#let bifold-gray-1 = rgb("#666666")
#let bifold-gray-2 = rgb("#808080")
#let bifold-gray-3 = rgb("#999999")
#let bifold-gray-4 = rgb("#cccccc")
#let bifold-gray-5 = rgb("#e6e6e6")

#let title-size = 120pt
#let subtitle-size = 65pt
#let authors-size = 50pt
#let institutes-size = 30pt
#let heading-size = 48pt

// BIFOLD poster template
#let bifold-a0-poster(doc) = [
  #set page(
    "a0",
    margin: (top: 3cm, left: 4cm, right: 4cm, bottom: 2cm),
    background: image("logos/BIFOLD_bg.png"),
  )
  #set text(
    font: ("Catamaran", "Helvetica", "Arial"),
    size: layout-a0.at("body-size"),
  )
  #set super(typographic: false) // otherwise look weird Catamaran

  #set cite(style: "springer-basic")

  #let box-spacing = 1.5cm
  #set columns(gutter: box-spacing)
  #set block(spacing: box-spacing)
  #set-poster-layout(layout-a0)
  #set-theme((
    "heading-box-args": (
      inset: 0.6em,
      width: 100%,
      fill: bifold-blue-6,
      stroke: bifold-blue-6,
    ),
    "heading-text-args": (
      fill: bifold-blue-1,
      weight: "bold",
    ),
    "body-box-args": (
      inset: 0.6em,
      width: 100%,
      fill: white,
      stroke: white,
    ),
    "body-text-args": (:),
  ))
  #update-poster-layout(
    spacing: box-spacing,
    title-size: title-size,
    subtitle-size: subtitle-size,
    heading-size: heading-size,
  )
  #doc
]

#let bifold-title-box(
  title,
  subtitle: none,
  authors: none,
  institutes: none,
  keywords: none,
  logo: none,
  background: none,
  text-relative-width: 80%,
  spacing: 5%,
  title-size: none,
  subtitle-size: none,
  authors-size: none,
  institutes-size: none,
  keywords-size: none,
) = {
  context {
    let text-relative-width = text-relative-width
    /// Get theme and layout state
    let pl = _state-poster-layout.at(here())

    /// Layout specific options
    let title-size = if title-size == none { pl.at("title-size") } else {
      title-size
    }
    let subtitle-size = if subtitle-size == none {
      pl.at("subtitle-size")
    } else { subtitle-size }
    let authors-size = if authors-size == none { pl.at("authors-size") } else {
      authors-size
    }
    let institutes-size = if institutes-size == none {
      pl.at("institutes-size")
    } else {
      institutes-size
    }
    let keywords-size = if keywords-size == none {
      pl.at("keywords-size")
    } else { keywords-size }

    /// Generate body of box
    let text-content = grid(
      columns: (8fr, 2fr),
      align: (horizon + left, horizon + right),
      column-gutter: 0em,
      [
        #set par(leading: 0.6em, spacing: 0.6em)
        #set text(size: subtitle-size)
        #if subtitle != none { [#subtitle] }
        #set text(size: authors-size)
        #if authors != none { [#authors] }
        #if institutes != none {
          [
            #set text(size: institutes-size, weight: "regular")
            #par(institutes, leading: 0.75em)
          ]
        }
        #if keywords != none {
          [
            #v(1em, weak: true)
            #set text(size: keywords-size)
            #keywords
          ]
        }
      ],
      [
        #image("iclr_qr.png", width: 60%)
      ],
    )

    /// Expand to full width of no image is specified
    if logo == none {
      text-relative-width = 100%
    }

    /// Big title
    [#set text(size: title-size, fill: white, weight: "extrabold")
      #upper(title)
    ]
    v(2.5em, weak: true)
    /// Finally construct the main rectangle
    common-box(
      heading: [
        #background
        // #v(-measure(background).height)
        #text-content
      ],
      heading-box-args: (
        inset: 1em,
        width: 100%,
        fill: white,
        stroke: bifold-blue-6,
      ),
    )
  }
}
