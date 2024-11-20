using DifferentiationInterface
using SparseConnectivityTracer, SparseMatrixColorings
using ForwardDiff: ForwardDiff

iter_diff(x, k) = k == 0 ? x : diff(iter_diff(x, k - 1))

dense_backend = AutoForwardDiff()

sparsity_detector = TracerSparsityDetector()
coloring_algorithm = GreedyColoringAlgorithm()
sparse_backend = AutoSparse(dense_backend; sparsity_detector, coloring_algorithm)

x, k = rand(10), 3;
jacobian(iter_diff, dense_backend, x, Constant(k))
jacobian(iter_diff, sparse_backend, x, Constant(k))

prep = prepare_jacobian(iter_diff, sparse_backend, x, Constant(k));

jacobian(iter_diff, prep, sparse_backend, x, Constant(k))

ncolors(prep)
sparsity_pattern(prep)

column_colors(prep)

using DifferentiationInterfaceTest

scen = Scenario{:jacobian,:out}(iter_diff, rand(1000); contexts=(Constant(10),))
data = benchmark_differentiation([dense_backend, sparse_backend], [scen]; benchmark=:full)
