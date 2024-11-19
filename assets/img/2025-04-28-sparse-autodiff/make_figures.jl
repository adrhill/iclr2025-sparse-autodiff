using Pkg
Pkg.activate(@__DIR__)
using Luxor
using Colors
using MathTeXEngine
using LaTeXStrings: LaTeXString
using StableRNGs: StableRNG

using Flux: Conv, relu
using DifferentiationInterface: jacobian, AutoForwardDiff
using ForwardDiff: ForwardDiff
using LinearAlgebra: I
using SparseArrays

# https://github.com/MakieOrg/Makie.jl/blob/e90c042d16b461e67b750e5ce53790e732281dba/src/theming.jl#L1-L16
# Conservative 7-color palette from Points of view: Color blindness, Bang Wong - Nature Methods
# https://www.nature.com/articles/nmeth.1618?WT.ec_id=NMETH-201106
COLORSCHEME_WONG =
    convert.(
        HSL,
        (
            RGB(0 / 255, 114 / 255, 178 / 255),   # blue
            RGB(230 / 255, 159 / 255, 0 / 255),   # orange
            RGB(0 / 255, 158 / 255, 115 / 255),   # green
            RGB(204 / 255, 121 / 255, 167 / 255), # reddish purple
            RGB(86 / 255, 180 / 255, 233 / 255),  # sky blue
            RGB(213 / 255, 94 / 255, 0 / 255),    # vermillion
            RGB(240 / 255, 228 / 255, 66 / 255),  # yellow
        ),
    )

get_hue(c::HSL) = hue(c)
get_hue(c::Colorant) = get_hue(convert(HSL, c))

blue, orange, green, purple, lightblue, vermillion, yellow = COLORSCHEME_WONG
hue_blue, hue_orange, hue_green, hue_purple, hue_lightblue, hue_vermillion, hue_yellow =
    get_hue.(COLORSCHEME_WONG)

lightness = 0.4
saturation = 0.8

color_black = convert(HSL, Gray(0))
color_white = convert(HSL, Gray(1))
color_operator = convert(HSL, Gray(0.3))
color_transparent = RGBA(0, 0, 0, 0)
color_background = color_transparent

named_color(name) = convert(HSL, RGB(Colors.color_names[name] ./ 256...))

color_F = blue
color_H = orange
color_G = green
color_vector = purple

# Colors for matrix coloring
mc1 = yellow
mc2 = lightblue
mc3 = vermillion # used for suboptimal coloring

CELLSIZE = 20
PADDING = 2  # Increased PADDING for better spacing
FONTSIZE = 18
SPACE = 11

# Function to normalize value between 0 and 1
normalize(x, min, max) = (x - min) / (max - min)
scale(x, min, max, lo, hi) = normalize(x, min, max) * (hi - lo) + lo

abstract type Drawable end
width(D::Drawable) = D.width
height(D::Drawable) = D.height

struct Position{D<:Drawable}
    drawable::D
    center::Point
end
drawable(P::Position) = P.drawable
width(P::Position) = width(drawable(P))
height(P::Position) = height(drawable(P))

center(P::Position) = P.center
xcenter(P::Position) = center(P).x
ycenter(P::Position) = center(P).y

top(P::Position) = Point(xcenter(P), ycenter(P) - height(P) / 2)
bottom(P::Position) = Point(xcenter(P), ycenter(P) + height(P) / 2)
right(P::Position) = Point(xcenter(P) + width(P) / 2, ycenter(P))
left(P::Position) = Point(xcenter(P) - width(P) / 2, ycenter(P))

function position_right_of(P::Position; space = SPACE)
    x, y = right(P)
    function position_drawable(D::Drawable)
        return Position(D, Point(x + space + width(D) / 2, y))
    end
    return position_drawable
end

function position_above(P::Position; space = SPACE)
    x, y = top(P)
    function position_drawable(D::Drawable)
        return Position(D, Point(x, y - space - height(D) / 2))
    end
    return position_drawable
end

function position_on(P::Position)
    return function position_drawable(D::Drawable)
        return Position(D, center(P))
    end
end

function draw!(P::Position; offset = Point(0.0, 0.0))
    center = P.center + offset
    draw!(P.drawable, center)
    return nothing
end

#========#
# Matrix #
#========#

default_cell_text(x) = string(round(x; digits = 2))
Base.@kwdef struct DrawMatrix <: Drawable
    mat::Matrix{Float64}
    color::HSL{Float64} = color_black
    cellsize::Float64 = CELLSIZE
    padding_inner::Float64 = PADDING
    padding_outer::Float64 = 1.75 * PADDING
    border_inner::Float64 = 0.75
    border_outer::Float64 = 2.0
    dashed::Bool = false
    show_text::Bool = false
    mat_text::Matrix{String} = map(default_cell_text, mat)
    mat_colors = fill(color, size(mat))
    absmax::Float64 = maximum(abs, mat)
    height::Float64 =
        size(mat, 1) * (cellsize + padding_inner) - padding_inner + 2 * padding_outer
    width::Float64 =
        size(mat, 2) * (cellsize + padding_inner) - padding_inner + 2 * padding_outer
end

function draw!(M::DrawMatrix, center::Point)
    # Destructure DrawMatrix for convenience
    (;
        mat,
        color,
        cellsize,
        padding_inner,
        padding_outer,
        border_inner,
        border_outer,
        dashed,
        show_text,
        mat_text,
        mat_colors,
        absmax,
        height,
        width,
    ) = M

    rows, cols = size(mat)

    # Apply offset
    xcenter, ycenter = center
    # Compute upper left edge of matrix
    x0 =
        xcenter - (cols / 2) * (cellsize + padding_inner) + padding_inner / 2 -
        padding_outer
    y0 =
        ycenter - (rows / 2) * (cellsize + padding_inner) + padding_inner / 2 -
        padding_outer

    setline(1)
    for i = 1:rows
        for j = 1:cols
            # Calculate cell position (corner of matrix entry)
            x = x0 + (j - 1) * (cellsize + padding_inner) + padding_outer
            y = y0 + (i - 1) * (cellsize + padding_inner) + padding_outer

            # Calculate color based on normalized value
            val = mat[i, j]
            cell_color = convert(HSL, mat_colors[i, j])
            (; h, s, l) = cell_color
            l_new = iszero(val) ? 1.0 : l * scale(abs(val), 0, absmax, 1.65, 0.65)
            cell_color_background = HSL(h, s, l_new)

            # Draw rectangle
            setcolor(cell_color_background)
            rect(Point(x, y), cellsize, cellsize, :fill)

            # Draw border
            setline(border_inner)
            setcolor(cell_color)
            iszero(val) && setcolor("lightgray")
            rect(Point(x, y), cellsize, cellsize, :stroke)

            # Add text showing matrix value
            if show_text
                fontsize(min(cellsize ÷ 3, 14))
                if luma(cell_color_background) > 0.6
                    setcolor(HSL(h, s, 0.15)) # dark
                else
                    setcolor(HSL(h, s, 0.95)) # bright
                end
                iszero(val) && setcolor("lightgray")
                text(
                    mat_text[i, j],
                    Point(x + cellsize / 2, y + cellsize / 2);
                    halign = :center,
                    valign = :middle,
                )
            end
        end
    end

    # Draw border
    setline(border_outer)
    setcolor(color)
    dashed && setdash([7.0, 4.0])
    setlinejoin("miter")
    rect(Point(x0, y0), width, height, :stroke)
    return setdash("solid")
end

luma(c::Colorant) = luma(convert(RGB, c))
luma(c::RGB) = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b # using BT. 709 coefficients

#==========#
# Operator #
#==========#

Base.@kwdef struct DrawText <: Drawable
    text::LaTeXString
    color::HSL{Float64} = color_operator
    cellsize::Float64 = 12
    fontsize::Float64 = 20
end
width(O::DrawText) = O.cellsize
height(O::DrawText) = O.cellsize

function draw!(O::DrawText, center)
    # Apply offset
    setcolor(O.color)
    fontsize(O.fontsize)
    return text(
        O.text,
        center - Point(0.15 * O.cellsize, 0.0);
        halign = :center,
        valign = :middle,
    )
end

#==========#
# Operator #
#==========#

Base.@kwdef struct DrawOverlay <: Drawable
    text::LaTeXString
    color::HSL{Float64} = color_operator
    background::HSL{Float64} = HSL(color.h, color.s, 0.9)
    width::Float64 = 50
    height::Float64 = 33
    radius::Float64 = 12
    fontsize::Float64 = 20
end

function draw!(O::DrawOverlay, center)
    setline(0.75)
    fontsize(O.fontsize)
    # Draw box
    setcolor(O.background)
    box(center, O.width, O.height, O.radius; action = :fill)
    # Draw darker outline
    setcolor(HSL(O.color.h, O.color.s, 0.3))
    box(center, O.width, O.height, O.radius; action = :stroke)
    return text(O.text, center + Point(0, 2); halign = :center, valign = :middle)
end

#======#
# Node #
#======#

Base.@kwdef struct DrawNode <: Drawable
    text::LaTeXString
    color::HSL{Float64} = color_operator
    background::HSL{Float64} = HSL(color.h, color.s, 0.9)
    radius::Float64 = 16
    fontsize::Float64 = 20
end

function draw!(O::DrawNode, center)
    setline(0.75)
    fontsize(O.fontsize)
    # Draw circle
    setcolor(O.background)
    circle(center, O.radius; action = :fill)
    # Draw darker outline
    setcolor(HSL(O.color.h, O.color.s, 0.3))
    circle(center, O.radius; action = :stroke)
    return text(O.text, center + Point(-1.25, 0); halign = :center, valign = :middle)
end

#==========#
# Draw PDF #
#==========#

# Get random matrices
m, n, p = 4, 5, 3
H = randn(StableRNG(121), m, p)
G = randn(StableRNG(123), p, n)
F = H * G

S = Matrix(
    [
        0.0 1.852 0.0 2.207 0.0
        0.0 0.0 0.0 0.970 -2.195
        0.0 -0.579 1.472 0.0 0.0
        -1.91 0.0 -0.464 0.0 0.0
    ],
)
iszero_string(x) = !iszero(x) ? "≠ 0" : "0"

# Pattern
P = map(!iszero, S)
P_text = map(iszero_string, P)

# Colors for pattern above
column_colors = [
    mc1 mc1 mc2 mc2 mc1
    mc1 mc1 mc2 mc2 mc1
    mc1 mc1 mc2 mc2 mc1
    mc1 mc1 mc2 mc2 mc1
]
row_colors = [
    mc1 mc1 mc1 mc1 mc1
    mc2 mc2 mc2 mc2 mc2
    mc2 mc2 mc2 mc2 mc2
    mc1 mc1 mc1 mc1 mc1
]

# Forward mode
vFfw = randn(StableRNG(3), n, 1)
vHfw = G * vFfw
vRfw = H * vHfw # result from right

# Reverse mode
vFrv = randn(StableRNG(3), 1, m)
vGrv = vFrv * H
vRrv = vGrv * G # result from left

# Create drawables
DF = DrawMatrix(; mat = F, color = color_F)
DG = DrawMatrix(; mat = G, color = color_G)
DH = DrawMatrix(; mat = H, color = color_H)

DFd = DrawMatrix(; mat = F, color = color_F, dashed = true)
DGd = DrawMatrix(; mat = G, color = color_G, dashed = true)
DHd = DrawMatrix(; mat = H, color = color_H, dashed = true)

DFdn = DrawMatrix(; mat = F, color = color_F, dashed = true, show_text = true)

DEq = DrawText(; text = "=")
DTimes = DrawText(; text = "⋅")
DCirc = DrawText(; text = "∘")

DJF = DrawOverlay(; text = L"J_{f}(x)", color = color_F)
DJG = DrawOverlay(; text = L"J_{g}(x)", color = color_G)
DJH = DrawOverlay(; text = L"J_{h}(g(x))", color = color_H, fontsize = 18, width = 65)

DDF = DrawOverlay(; text = "Df(x)", color = color_F, fontsize = 18)
DDG = DrawOverlay(; text = "Dg(x)", color = color_G, fontsize = 18)
DDH = DrawOverlay(; text = "Dh(g(x))", color = color_H, fontsize = 15, width = 65)

function setup!()
    background(color_background)
    fontsize(18)
    fontface("JuliaMono")
end

function chainrule(; show_text = true)
    setup!()

    DFn = DrawMatrix(; mat = F, color = color_F, show_text = show_text)
    DGn = DrawMatrix(; mat = G, color = color_G, show_text = show_text)
    DHn = DrawMatrix(; mat = H, color = color_H, show_text = show_text)

    # Position drawables
    drawables = [DFn, DEq, DHn, DTimes, DGn]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(DF) - total_width) / 2
    ystart = 0.0

    PF = Position(DFn, Point(xstart, ystart))
    PEq = position_right_of(PF)(DEq)
    PH = position_right_of(PEq)(DHn)
    PTimes = position_right_of(PH)(DTimes)
    PG = position_right_of(PTimes)(DGn)

    PJF = position_on(PF)(DJF)
    PJG = position_on(PG)(DJG)
    PJH = position_on(PH)(DJH)

    # Draw 
    for obj in (PF, PG, PH, PEq, PTimes, PJF, PJG, PJH)
        draw!(obj)
    end
end

# Draw the Jacobian of the first layer of the small LeNet-5 CNN
function big_conv_jacobian()
    setup!()
    layer = Conv((5, 5), 1 => 1, identity)
    input = randn(Float32, 28, 28, 1, 1)

    J = jacobian(layer, AutoForwardDiff(), input)
    # @info "Size of the Conv Jacobian:" size(J) relative_sparsity=sum(iszero,J)/length(J)

    DJ = DrawMatrix(;
        mat = J,
        color = color_G,
        cellsize = 2,
        padding_inner = 0,
        padding_outer = 0,
        border_inner = 0,
        border_outer = 10,
    )
    DJF = DrawOverlay(;
        text = L"J_{g}(x)",
        color = color_G,
        fontsize = 150,
        width = 350,
        height = 180,
    )

    center = Point(0.0, 0.0)
    PJ = Position(DJ, center)
    PDJ = Position(DJF, center)
    draw!(PJ)
    return draw!(PDJ)
end

function matrixfree()
    setup!()

    # Position drawables
    drawables = [DFd, DEq, DHd, DCirc, DGd]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(DF) - total_width) / 2
    ystart = 0.0

    PF = Position(DFd, Point(xstart, ystart))
    PEq = position_right_of(PF)(DEq)
    PH = position_right_of(PEq)(DHd)
    PCirc = position_right_of(PH)(DCirc)
    PG = position_right_of(PCirc)(DGd)

    PDF = position_on(PF)(DDF)
    PDG = position_on(PG)(DDG)
    PDH = position_on(PH)(DDH)

    # Draw 
    for obj in (PF, PG, PH, PEq, PCirc, PDF, PDG, PDH)
        draw!(obj)
    end
end

function forward_mode_eval()
    setup!()

    # Create three vectors
    DvFfw = DrawMatrix(; mat = vFfw, color = color_vector) # input
    DvHfw = DrawMatrix(; mat = vHfw, color = color_vector)
    DvRfw = DrawMatrix(; mat = vRfw, color = color_vector) # output

    # Position drawables
    drawables = [DFd, DvFfw, DEq, DHd, DGd, DvFfw]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(DF) - total_width) / 2
    ystart = -105.0

    PF = Position(DFd, Point(xstart, ystart))
    PvFfw = position_right_of(PF)(DvFfw)

    PEq = position_right_of(PvFfw)(DEq)
    PH = position_right_of(PEq)(DHd)
    PG = position_right_of(PH)(DGd)
    PvFfw2 = position_right_of(PG)(DvFfw)

    PEq2 = Position(DEq, center(PEq) + Point(0, 110))
    PH2 = position_right_of(PEq2)(DHd)
    PvHfw = position_right_of(PH2)(DvHfw)

    PEq3 = Position(DEq, center(PEq2) + Point(0, 110))
    PvRfw = position_right_of(PEq3)(DvRfw)

    PDF = position_on(PF)(DDF)
    PDG = position_on(PG)(DDG)
    PDH = position_on(PH)(DDH)
    PDH2 = position_on(PH2)(DDH)

    # Draw 
    for obj in
        (PF, PvFfw, PEq, PH, PG, PvFfw2, PEq2, PH2, PvHfw, PEq3, PvRfw, PDF, PDG, PDH, PDH2)
        draw!(obj)
    end
end

function reverse_mode_eval()
    setup!()

    # Create drawables for three vectors
    DvFrv = DrawMatrix(; mat = vFrv, color = color_vector) # input
    DvGrv = DrawMatrix(; mat = vGrv, color = color_vector)
    DvRrv = DrawMatrix(; mat = vRrv, color = color_vector) # output

    # Position drawables
    drawables = [DvFrv, DFd, DEq, DvFrv, DHd, DGd]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(DvFrv) - total_width) / 2
    ystart = -82.5

    PvFrv = Position(DvFrv, Point(xstart, ystart))
    PF = position_right_of(PvFrv)(DFd)

    PEq = position_right_of(PF)(DEq)
    PvFrv2 = position_right_of(PEq)(DvFrv)
    PH = position_right_of(PvFrv2)(DHd)
    PG = position_right_of(PH)(DGd)

    PEq2 = Position(DEq, center(PEq) + Point(0, 110))
    PvGrv = position_right_of(PEq2)(DvGrv)
    PG2 = position_right_of(PvGrv)(DGd)

    PEq3 = Position(DEq, center(PEq2) + Point(0, 90))
    PvRrv = position_right_of(PEq3)(DvRrv)

    PDF = position_on(PF)(DDF)
    PDG = position_on(PG)(DDG)
    PDH = position_on(PH)(DDH)
    PDG2 = position_on(PG2)(DDG)

    # Draw 
    for obj in
        (PF, PvFrv, PEq, PH, PG, PvFrv2, PEq2, PG2, PvGrv, PEq3, PvRrv, PDF, PDG, PDH, PDG2)
        draw!(obj)
    end
end

function forward_mode()
    setup!()

    i1 = 1
    e1 = zeros(n, 1)
    e1[i1, 1] = 1

    i2 = 5
    e2 = zeros(n, 1)
    e2[i2, 1] = 1

    absmax = maximum(abs, F)

    F1_text = map(default_cell_text, F)
    F2_text = map(default_cell_text, F)
    F1_text[:, 2:end] .= ""
    F2_text[:, begin:(end-1)] .= ""

    DF1 = DrawMatrix(;
        mat = F,
        mat_text = F1_text,
        color = color_F,
        dashed = true,
        show_text = true,
    )
    DF2 = DrawMatrix(;
        mat = F,
        mat_text = F2_text,
        color = color_F,
        dashed = true,
        show_text = true,
    )
    De1 = DrawMatrix(; mat = e1, color = color_vector, show_text = true)
    De2 = DrawMatrix(; mat = e2, color = color_vector, show_text = true)
    DFe1 = DrawMatrix(; mat = F * e1, color = color_F, absmax = absmax, show_text = true)
    DFe2 = DrawMatrix(; mat = F * e2, color = color_F, absmax = absmax, show_text = true)
    DDots = DrawText(; text = "...", fontsize = 40)

    # Position drawables
    drawables = [DF1, De1, DEq, DFe1, DDots, DFdn, De2, DEq, DFe2]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE + 40
    xstart = (width(DF1) - total_width) / 2
    ystart = 0.0

    PF1 = Position(DF1, Point(xstart, ystart))
    Pe1 = position_right_of(PF1)(De1)
    PEq1 = position_right_of(Pe1)(DEq)
    PFe1 = position_right_of(PEq1)(DFe1)

    PDots = position_right_of(PFe1; space = 30)(DDots)

    PF2 = position_right_of(PDots; space = 30)(DF2)
    Pe2 = position_right_of(PF2)(De2)
    PEq2 = position_right_of(Pe2)(DEq)
    PFe2 = position_right_of(PEq2)(DFe2)

    PDF1 = position_on(PF1)(DDF)
    PDF2 = position_on(PF2)(DDF)

    # Draw 
    for obj in (PF1, Pe1, PEq1, PFe1, PF2, Pe2, PEq2, PFe2, PDF1, PDF2, PDots)
        draw!(obj)
    end
end

function reverse_mode()
    setup!()

    i1 = 1
    e1 = zeros(1, m)
    e1[1, i1] = 1

    i2 = 4
    e2 = zeros(1, m)
    e2[1, i2] = 1

    absmax = maximum(abs, F)

    F1_text = map(default_cell_text, F)
    F2_text = map(default_cell_text, F)
    F1_text[2:end, :] .= ""
    F2_text[begin:(end-1), :] .= ""

    DF1 = DrawMatrix(;
        mat = F,
        mat_text = F1_text,
        color = color_F,
        dashed = true,
        show_text = true,
    )
    DF2 = DrawMatrix(;
        mat = F,
        mat_text = F2_text,
        color = color_F,
        dashed = true,
        show_text = true,
    )

    De1 = DrawMatrix(; mat = e1, color = color_vector, show_text = true)
    De2 = DrawMatrix(; mat = e2, color = color_vector, show_text = true)
    DFe1 = DrawMatrix(; mat = e1 * F, color = color_F, absmax = absmax, show_text = true)
    DFe2 = DrawMatrix(; mat = e2 * F, color = color_F, absmax = absmax, show_text = true)
    DDots = DrawText(; text = "...", fontsize = 40)

    # Position drawables
    drawables = [De1, DF1, DEq, DFe1]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(De1) - total_width) / 2
    ystart = -65.0

    Pe1 = Position(De1, Point(xstart, ystart))
    PF1 = position_right_of(Pe1)(DF1)
    PEq1 = position_right_of(PF1)(DEq)
    PFe1 = position_right_of(PEq1)(DFe1)

    PDots = Position(DDots, Point(0, ystart + 71.0))

    Pe2 = Position(De2, Point(xstart, ystart + 140.0))
    PF2 = position_right_of(Pe2)(DF2)
    PEq2 = position_right_of(PF2)(DEq)
    PFe2 = position_right_of(PEq2)(DFe2)

    PDF1 = position_on(PF1)(DDF)
    PDF2 = position_on(PF2)(DDF)

    # Draw 
    for obj in (PF1, Pe1, PEq1, PFe1, PF2, Pe2, PEq2, PFe2, PDF1, PDF2, PDots)
        draw!(obj)
    end
end

function sparsity(; ismap = false)
    setup!()
    DS = DrawMatrix(; mat = S, color = color_F, dashed = ismap, show_text = !ismap)
    PS = Position(DS, Point(0.0, 0.0))
    return draw!(PS)
end

function sparse_map_colored()
    setup!()

    DS = DrawMatrix(;
        mat = S,
        color = color_F,
        dashed = true,
        show_text = true,
        mat_colors = column_colors,
    )
    PS = Position(DS, Point(0.0, 0.0))
    return draw!(PS)
end

function sparsity_pattern()
    setup!()

    P_text = map(x -> !iszero(x) ? "≠ 0" : "0", P)
    DP = DrawMatrix(; mat = P, mat_text = P_text, color = color_F, show_text = true)
    PP = Position(DP, Point(0.0, 0.0))
    return draw!(PP)
end

function sparsity_pattern_representations()
    setup!()

    # Dense Jacobian
    DS = DrawMatrix(; mat = S, color = color_F, dashed = false, show_text = true)

    # Binary Jacobian
    B_text = map(x -> !iszero(x) ? "≠ 0" : "0", P)
    DB = DrawMatrix(; mat = P, mat_text = B_text, color = color_F, show_text = true)

    # Index set representations
    I = fill(1.0, m, 1)
    I_text = reshape(["{2,4}", "{4,5}", "{2,3}", "{1,3}"], m, 1)
    DI = DrawMatrix(; mat = I, mat_text = I_text, color = color_F, show_text = true)

    # Text labels
    fontsize = 11
    Da = DrawText(; text = "(a)", fontsize)
    Db = DrawText(; text = "(b)", fontsize)
    Dc = DrawText(; text = "(c)", fontsize)

    # Center drawables
    space = 30
    drawables = [DS, DB, DI]
    total_width = sum(width, drawables) + (length(drawables) - 1) * space
    xstart = (width(DS) - total_width) / 2

    PS = Position(DS, Point(xstart, 7.5))
    PB = position_right_of(PS; space)(DB)
    PI = position_right_of(PB; space)(DI)

    Pa = position_above(PS; space = 7)(Da)
    Pb = position_above(PB; space = 7)(Db)
    Pc = position_above(PI; space = 7)(Dc)

    for obj in (PS, PB, PI, Pa, Pb, Pc)
        draw!(obj)
    end
end

function sparsity_coloring()
    setup!()

    DP = DrawMatrix(;
        mat = P,
        mat_text = P_text,
        color = color_F,
        show_text = true,
        mat_colors = column_colors,
    )
    PP = Position(DP, Point(0.0, 0.0))
    return draw!(PP)
end

function sparse_ad()
    setup!()

    v = reshape([1.0 1.0 0.0 0.0 1.0], n, 1)
    absmax = maximum(abs, S)

    DS = DrawMatrix(;
        mat = S,
        color = color_F,
        dashed = true,
        show_text = true,
        mat_colors = column_colors,
    )
    Dv = DrawMatrix(; mat = v, color = color_vector, show_text = true)
    DSv = DrawMatrix(;
        mat = S * v,
        color = color_F,
        absmax = absmax,
        show_text = true,
        mat_colors = fill(mc1, n, 1),
    )

    # Position drawables
    drawables = [DS, Dv, DEq, DSv]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(DS) - total_width) / 2
    ystart = 0.0

    PS = Position(DS, Point(xstart, ystart))
    Pv = position_right_of(PS)(Dv)
    PEq = position_right_of(Pv)(DEq)
    PSv = position_right_of(PEq)(DSv)

    # Draw 
    for obj in (PS, Pv, PEq, PSv)
        draw!(obj)
    end
end

function sparse_ad_forward_full()
    setup!()

    v1 = reshape([1.0 1.0 0.0 0.0 1.0], n, 1)
    v2 = reshape([0.0 0.0 1.0 1.0 0.0], n, 1)
    absmax = maximum(abs, S)

    DS = DrawMatrix(;
        mat = S,
        color = color_F,
        dashed = true,
        show_text = true,
        mat_colors = column_colors,
    )
    Dv1 = DrawMatrix(; mat = v1, color = color_vector, show_text = true)
    Dv2 = DrawMatrix(; mat = v2, color = color_vector, show_text = true)
    DSv1 = DrawMatrix(;
        mat = S * v1,
        color = color_F,
        absmax = absmax,
        show_text = true,
        mat_colors = fill(mc1, n, 1),
    )
    DSv2 = DrawMatrix(;
        mat = S * v2,
        color = color_F,
        absmax = absmax,
        show_text = true,
        mat_colors = fill(mc2, n, 1),
    )

    ## Position drawables
    drawables = [DS, Dv1, DEq, DSv1]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(DS) - total_width) / 2
    ystart = -65.0

    # First row
    PS1 = Position(DS, Point(xstart, ystart))
    Pv1 = position_right_of(PS1)(Dv1)
    PEq1 = position_right_of(Pv1)(DEq)
    PSv1 = position_right_of(PEq1)(DSv1)

    # Second row
    PS2 = Position(DS, Point(xstart, -ystart))
    Pv2 = position_right_of(PS2)(Dv2)
    PEq2 = position_right_of(Pv2)(DEq)
    PSv2 = position_right_of(PEq2)(DSv2)

    # Draw 
    for obj in (PS1, Pv1, PEq1, PSv1, PS2, Pv2, PEq2, PSv2)
        draw!(obj)
    end
end

function sparse_ad_forward_decompression()
    setup!()

    v1 = reshape([1.0 1.0 0.0 0.0 1.0], n, 1)
    v2 = reshape([0.0 0.0 1.0 1.0 0.0], n, 1)
    absmax = maximum(abs, S)

    DS = DrawMatrix(;
        mat = S,
        color = color_F,
        dashed = false,
        show_text = true,
        mat_colors = column_colors,
    )
    DSv1 = DrawMatrix(;
        mat = S * v1,
        color = color_F,
        absmax = absmax,
        show_text = true,
        mat_colors = fill(mc1, m, 1),
    )
    DSv2 = DrawMatrix(;
        mat = S * v2,
        color = color_F,
        absmax = absmax,
        show_text = true,
        mat_colors = fill(mc2, m, 1),
    )
    DArrow = DrawText(; text = "→")

    ## Position drawables
    drawables = [DSv1, DSv2, DEq, DS]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(DSv1) - total_width) / 2
    ystart = 0

    PSv1 = Position(DSv1, Point(xstart, ystart))
    PSv2 = position_right_of(PSv1; space = 5)(DSv2)
    PArrow = position_right_of(PSv2)(DArrow)
    PS = position_right_of(PArrow; space = 16)(DS)

    # Draw 
    for obj in (PSv1, PSv2, PArrow, PS)
        draw!(obj)
    end
end

function sparse_ad_reverse_full()
    setup!()

    v1 = reshape([1.0 0.0 0.0 1.0], 1, m)
    v2 = reshape([0.0 1.0 1.0 0.0], 1, m)
    absmax = maximum(abs, S)

    DS = DrawMatrix(;
        mat = S,
        color = color_F,
        dashed = true,
        show_text = true,
        mat_colors = row_colors,
    )
    Dv1 = DrawMatrix(; mat = v1, color = color_vector, show_text = true)
    Dv2 = DrawMatrix(; mat = v2, color = color_vector, show_text = true)
    DSv1 = DrawMatrix(;
        mat = v1 * S,
        color = color_F,
        absmax = absmax,
        show_text = true,
        mat_colors = fill(mc1, 1, n),
    )
    DSv2 = DrawMatrix(;
        mat = v2 * S,
        color = color_F,
        absmax = absmax,
        show_text = true,
        mat_colors = fill(mc2, 1, n),
    )

    ## Position drawables
    drawables = [Dv1, DS, DEq, DSv1]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(Dv1) - total_width) / 2
    ystart = -60.0

    # First row
    Pv1 = Position(Dv1, Point(xstart, ystart))
    PS1 = position_right_of(Pv1)(DS)
    PEq1 = position_right_of(PS1)(DEq)
    PSv1 = position_right_of(PEq1)(DSv1)

    # Second row
    Pv2 = Position(Dv2, Point(xstart, -ystart))
    PS2 = position_right_of(Pv2)(DS)
    PEq2 = position_right_of(PS2)(DEq)
    PSv2 = position_right_of(PEq2)(DSv2)

    # Draw 
    for obj in (PS1, Pv1, PEq1, PSv1, PS2, Pv2, PEq2, PSv2)
        draw!(obj)
    end
end

function sparse_ad_reverse_decompression()
    setup!()

    v1 = reshape([1.0 0.0 0.0 1.0], 1, m)
    v2 = reshape([0.0 1.0 1.0 0.0], 1, m)
    absmax = maximum(abs, S)

    DS = DrawMatrix(;
        mat = S,
        color = color_F,
        dashed = false,
        show_text = true,
        mat_colors = row_colors,
    )
    DSv1 = DrawMatrix(;
        mat = v1 * S,
        color = color_F,
        absmax = absmax,
        show_text = true,
        mat_colors = fill(mc1, 1, n),
    )
    DSv2 = DrawMatrix(;
        mat = v2 * S,
        color = color_F,
        absmax = absmax,
        show_text = true,
        mat_colors = fill(mc2, 1, n),
    )
    DArrow = DrawText(; text = "→")

    ## Position drawables
    drawables = [DSv1, DEq, DS]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(DSv1) - total_width) / 2
    ystart = -16

    PSv1 = Position(DSv1, Point(xstart, ystart))
    PSv2 = Position(DSv2, Point(xstart, -ystart))
    xarrow = right(PSv1).x + SPACE + width(DArrow) / 2
    PArrow = Position(DArrow, Point(xarrow, 0))
    PS = position_right_of(PArrow; space = 16)(DS)

    # Draw 
    for obj in (PSv1, PSv2, PArrow, PS)
        draw!(obj)
    end
end

function forward_mode_naive()
    setup!()

    DFd = DrawMatrix(; mat = S, color = color_F, dashed = true, show_text = false)
    DI = DrawMatrix(; mat = I(5), color = color_vector, show_text = true)
    DFj = DrawMatrix(; mat = S * I(5), color = color_F, show_text = true)

    # Position drawables
    drawables = [DFd, DI, DEq, DFj]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(DFd) - total_width) / 2
    ystart = 0.0

    PFd = Position(DFd, Point(xstart, ystart))
    PI = position_right_of(PFd)(DI)
    PEq = position_right_of(PI)(DEq)
    PFj = position_right_of(PEq)(DFj)
    @info PFd.center # Point(-137.5, 0.0)

    PDF = position_on(PFd)(DDF)
    PJF = position_on(PFj)(DJF)

    # Draw 
    for obj in (PFd, PI, PEq, PFj, PDF, PJF)
        draw!(obj)
    end
end

function forward_mode_sparse()
    setup!()

    PI = fill(1, n, 1)
    PJ = fill(1, m, 1)

    PI_text = reshape(["{1}", "{2}", "{3}", "{4}", "{5}"], n, 1)
    PJ_text = reshape(["{2,4}", "{4,5}", "{2,3}", "{1,3}"], m, 1)
    P_text = map(x -> !iszero(x) ? "≠ 0" : "0", P)

    DFd = DrawMatrix(; mat = S, color = color_F, dashed = true, show_text = false)
    DI = DrawMatrix(; mat = PI, mat_text = PI_text, color = color_vector, show_text = true)
    DFj = DrawMatrix(; mat = PJ, mat_text = PJ_text, color = color_F, show_text = true)
    DP = DrawMatrix(; mat = P, mat_text = P_text, color = color_F, show_text = true)

    DEq2 = DrawText(; text = "≔")

    # Position drawables
    PFd = Position(DFd, Point(-137.5, 0.0)) # reuse center from `forward_mode_naive`
    PI = position_right_of(PFd)(DI)
    PEq1 = position_right_of(PI)(DEq)
    PFj = position_right_of(PEq1)(DFj)
    PEq2 = position_right_of(PFj)(DEq2)
    PP = position_right_of(PEq2)(DP)

    PDF = position_on(PFd)(DDF)

    # Draw 
    for obj in (PFd, PI, PEq1, PFj, PEq2, PP, PDF)
        draw!(obj)
    end
end


function colored_matrix()
    setup!()


end

function colored_graph(column_colors)
    setup!()

    column_colors = reshape(column_colors, 1, n)

    # Centers of vertices
    positions = ngon(Point(-85, 0), 40, 5, -2 * π / 5 - π / 2, vertices = true)

    # First we draw edges...
    setline(1)
    setcolor(color_operator)
    for r in eachrow(P)
        js = findall(!iszero, r)
        edges = Set((a, b) for a in js, b in js if a < b)
        for (a, b) in edges
            line(positions[a], positions[b])
            strokepath()
        end
    end

    # ... then vertices on top
    for (i, (c, pos)) in enumerate(zip(column_colors, positions))
        D = DrawNode(; text = string(i), radius = 14, fontsize = 16, color = c)
        P = Position(D, pos)
        draw!(P)
    end

    column_colors = repeat(column_colors, inner = (m, 1))

    DS = DrawMatrix(;
        mat = S,
        color = color_F,
        dashed = true,
        show_text = true,
        mat_colors = column_colors,
    )

    draw!(Position(DrawText(; text = "→"), Point(-10, 0)))
    draw!(Position(DS, Point(80, 0)))
end


# This one is huge, avoid SVG and PDF:
@png big_conv_jacobian() 1600 1200 joinpath(@__DIR__, "big_conv_jacobian")

# Change the default saving format here
var"@save" = var"@svg" # var"@pdf"

@save chainrule() 380 100 joinpath(@__DIR__, "chainrule")
@save chainrule(; show_text = true) 380 100 joinpath(@__DIR__, "chainrule_num")
@save matrixfree() 380 100 joinpath(@__DIR__, "matrixfree")

@save forward_mode_eval() 450 340 joinpath(@__DIR__, "forward_mode_eval")
@save reverse_mode_eval() 570 280 joinpath(@__DIR__, "reverse_mode_eval")

@save forward_mode() 510 120 joinpath(@__DIR__, "forward_mode")
@save reverse_mode() 380 250 joinpath(@__DIR__, "reverse_mode")

@save sparsity() 120 100 joinpath(@__DIR__, "sparse_matrix")
@save sparsity(; ismap = true) 120 100 joinpath(@__DIR__, "sparse_map")

@save sparse_ad() 220 120 joinpath(@__DIR__, "sparse_ad")
@save sparse_map_colored() 120 100 joinpath(@__DIR__, "sparse_map_colored")

@save sparsity_pattern() 120 100 joinpath(@__DIR__, "sparsity_pattern")
@save sparsity_pattern_representations() 330 120 joinpath(
    @__DIR__,
    "sparsity_pattern_representations",
)
@save sparsity_coloring() 120 100 joinpath(@__DIR__, "coloring")

# Sized need to match:
@save forward_mode_naive() 400 120 joinpath(@__DIR__, "forward_mode_naive")
@save forward_mode_sparse() 400 120 joinpath(@__DIR__, "forward_mode_sparse")

# Make sure the aspect ratios of these two figures match for the blog post layout
@save sparse_ad_forward_full() 230 260 joinpath(@__DIR__, "sparse_ad_forward_full")
@save sparse_ad_forward_decompression() 230 260 joinpath(
    @__DIR__,
    "sparse_ad_forward_decompression",
)

# Make sure the aspect ratios of these two figures match for the blog post layout
@save sparse_ad_reverse_full() 390 230 joinpath(@__DIR__, "sparse_ad_reverse_full")
@save sparse_ad_reverse_decompression() 290 170 joinpath(
    @__DIR__,
    "sparse_ad_reverse_decompression",
)

@save colored_graph([mc1 mc1 mc2 mc2 mc1]) 300 120 joinpath(@__DIR__, "colored_graph")
@save colored_graph([mc1 mc1 mc2 mc2 mc3]) 300 120 joinpath(
    @__DIR__,
    "colored_graph_suboptimal",
)
@save colored_graph([mc1 mc1 mc2 mc1 mc2]) 300 120 joinpath(
    @__DIR__,
    "colored_graph_infeasible",
)
