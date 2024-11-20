import DifferentiationInterface as DI
import SparseConnectivityTracer as SCT
import SparseMatrixColorings as SMC
import ForwardDiff as FD

function differences(x)
    n = length(x)
    y = similar(x, n - 1)
    for i in 1:(n - 1)
        y[i] = x[i + 1] - x[i]
    end
    return y
end

function iterated_differences(x, k)
    if k == 0
        return x
    else
        y = iterated_differences(x, k - 1)
        return differences(y)
    end
end

dense_backend = DI.AutoForwardDiff()

sparse_backend = DI.AutoSparse(
    dense_backend;
    sparsity_detector=SCT.TracerSparsityDetector(),
    coloring_algorithm=SMC.GreedyColoringAlgorithm(),
)

x, k = rand(10), 3;
DI.jacobian(iterated_differences, dense_backend, x, DI.Constant(k))
DI.jacobian(iterated_differences, sparse_backend, x, DI.Constant(k))

prep = DI.prepare_jacobian(iterated_differences, sparse_backend, x, DI.Constant(k));

J = DI.jacobian(
    iterated_differences,
    prep,  # note the preparation result
    sparse_backend,
    x,
    DI.Constant(k),
)

SMC.ncolors(prep)
SMC.sparsity_pattern(prep)

SMC.column_colors(prep)

import DifferentiationInterfaceTest as DIT

scen = DIT.Scenario{:jacobian,:out}(
    iterated_differences, rand(1000); contexts=(DI.Constant(10),)
)
data = DIT.benchmark_differentiation([dense_backend, sparse_backend], [scen]; benchmark=:full)
