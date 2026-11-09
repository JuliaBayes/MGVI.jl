# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Test
using MGVI

using Distributions
using LinearAlgebra
using Random
using AutoDiffOperators
using MatrixShapedOperators: MatrixShapedOperator
import DistributionsAD  # required for Zygote AD through Distributions
import ForwardDiff, Zygote

struct _ParamsOnlyDist <: ContinuousUnivariateDistribution
    θ::Float64
end
Distributions.params(d::_ParamsOnlyDist) = (d.θ,)

if !isdefined(Main, :ModelPolyfit)
    include("test_models/model_polyfit.jl")
    import .ModelPolyfit
end

if !isdefined(Main, :ModelFFTGP)
    include("test_models/model_fft_gp.jl")
    import .ModelFFTGP
end

Test.@testset "test_jacobians_consistency" begin
    let
        A = rand(4,6)
        f = let A=A; x -> (A*(x.^2).^2) end
        x = rand(6); l = rand(4); r = rand(6)
        J_ref = ForwardDiff.jacobian(f, x)
        
        _, J1 = @inferred with_jacobian(f, x, AbstractMatrix, ADSelector(ForwardDiff))
        _, J2 = @inferred with_jacobian(f, x, MatrixShapedOperator, ADSelector(Zygote))
        _, J3 = @inferred with_jacobian(f, x, MatrixShapedOperator, ADSelector(ForwardDiff))

        for J in (J1, J2, J3)
            @test @inferred(Matrix(J)) ≈ J_ref
            @test @inferred(J' * l) ≈ J_ref' * l
            @test @inferred(J * r) ≈ J_ref * r
        end
    end

    let
        A = UpperTriangular([1.0 2.0 3.0; 0.0 4.0 5.0; 0.0 0.0 6.0])
        y, pb = Zygote.pullback(MGVI.flat_params, A)
        @test y == [1.0, 2.0, 4.0, 3.0, 5.0, 6.0]
        Δy = collect(1.0:6.0)
        g_ref = ForwardDiff.gradient(m -> MGVI.flat_params(UpperTriangular(m))' * Δy, Matrix(A))
        @test pb(Δy)[1] ≈ g_ref
    end

    let
        f = x -> MGVI.flat_params(MvNormal(x[1:2], [exp(x[3]) 0.1; 0.1 exp(x[4])]))
        x = [0.3, -0.2, 0.4, 0.7]
        J_ref = ForwardDiff.jacobian(f, x)
        _, J = with_jacobian(f, x, MatrixShapedOperator, ADSelector(Zygote))
        @test Matrix(J) ≈ J_ref
        # the traced value and vjp (TuringDenseMvNormal path) must match the primal:
        y_traced, pb = Zygote.pullback(f, x)
        @test y_traced ≈ f(x)
        w = collect(1.0:length(y_traced))
        @test pb(w)[1] ≈ J_ref'w
    end

    # The flat parameters of a distribution are its parameters in order,
    # also for distributions that provide `params` only:
    for d in (Normal(0.3, 1.2), Poisson(2.5), MvNormal([0.1, -0.2], [1.0 0.1; 0.1 2.0]), _ParamsOnlyDist(2.5))
        @test MGVI.flat_params(d) == MGVI.flat_params(params(d))
    end

    # Products flatten to the concatenated component parameters (a product
    # of Normals is an MvNormal in Distributions, so it is not listed here):
    for d in (Poisson(2.5), MvNormal([0.1, -0.2], [1.0 0.1; 0.1 2.0]), _ParamsOnlyDist(2.5))
        @test MGVI.flat_params(product_distribution([d, d])) == vcat(MGVI.flat_params(d), MGVI.flat_params(d))
    end

    # Products flatten their components via the components' own flat_params
    # methods, here the ones of the DistributionsAD extension that Zygote
    # tracing of MvNormal leads to:
    for model in (
        x -> product_distribution([MvNormal(x[1:2], [exp(x[3]) 0.1; 0.1 exp(x[4])])]),
        x -> product_distribution([MvNormal(x[1:2], Diagonal(exp.(x[3:4])))]),
        x -> product_distribution([MvNormal(x[1:2], exp(x[3]) * I)]),
    )
        f = MGVI.flat_params ∘ model
        x = [0.3, -0.2, 0.4, 0.7]
        y_traced, pb = Zygote.pullback(f, x)
        @test y_traced ≈ f(x)
        w = collect(1.0:length(y_traced))
        @test pb(w)[1] ≈ ForwardDiff.jacobian(f, x)'w
    end

    # Reverse-mode AD through the parameters of a large product distribution
    # must stay cheap. Since Distributions 0.25.133, `params` is a fold that
    # Zygote differentiates per element, allocating over 100 KiB each, where
    # flattening the parameters directly takes about 1 KiB:
    let
        λ = rand(Xoshiro(12), 10_000) .+ 0.5
        f = λ -> sum(MGVI.flat_params(product_distribution(Poisson.(λ))))
        Zygote.gradient(f, λ)
        @test Zygote.gradient(f, λ)[1] == ones(length(λ))
        @test (@allocated Zygote.gradient(f, λ)) < 16 * 1024 * length(λ)
    end

    epsilon = 1E-5
    Random.seed!(145)

    _flat_model = MGVI.flat_params ∘ ModelPolyfit.model
    true_params = ModelPolyfit.true_params

    _, full_jac = @inferred with_jacobian(_flat_model, true_params, AbstractMatrix, ADSelector(ForwardDiff))
    _, fwdder_jac = @inferred with_jacobian(_flat_model, true_params, MatrixShapedOperator, ADSelector(ForwardDiff))
    _, fwdrevad_jac = @inferred with_jacobian(_flat_model, true_params, MatrixShapedOperator, ADSelector(Zygote))

    for i in 1:min(size(full_jac)...)
        vec = rand(size(full_jac, 2))
        @test norm(fwdder_jac*vec - full_jac*vec) < epsilon
        @test norm(fwdrevad_jac*vec - full_jac*vec) < epsilon
    end

    # DistributionsAD swaps distribution types under Zygote tracing, the
    # traced function and its vjp must still match the primal parametrization:
    let
        y_primal = _flat_model(true_params)
        y_traced, pb = Zygote.pullback(_flat_model, true_params)
        @test y_traced ≈ y_primal
        for i in (1, size(full_jac, 1) ÷ 2, size(full_jac, 1))
            w = zeros(size(full_jac, 1)); w[i] = 1
            @test pb(w)[1] ≈ full_jac'w atol=epsilon
        end
    end
end

Test.@testset "test_jacobians_fft_gp" begin
    # Jacobians through an FFT-based harmonic-space GP model: the jvp runs
    # forward-mode through the dual-number DHT method, the vjp reverse-mode
    # through the FFT rules:
    _flat_model = MGVI.flat_params ∘ ModelFFTGP.model
    p = ModelFFTGP.starting_point
    J_ref = ForwardDiff.jacobian(_flat_model, p)
    _, J = with_jacobian(_flat_model, p, MatrixShapedOperator, ADSelector(Zygote))
    l = rand(size(J_ref, 1))
    r = rand(size(J_ref, 2))
    @test J * r ≈ J_ref * r
    @test J' * l ≈ J_ref' * l
end
