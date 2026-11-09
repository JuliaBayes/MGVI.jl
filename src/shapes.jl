# This file is a part of MGVI.jl, licensed under the MIT License (MIT).


flat_params(x::Real) = _svector((x,))
flat_params(x::StaticVector{N,T}) where {N,T<:Real} = x
flat_params(x::NTuple{N,T}) where {N,T<:Real} = _svector(x)
flat_params(x::Tuple) = vcat(map(flat_params, x)...)
flat_params(x::NamedTuple) = flat_params(values(x))
flat_params(x::AbstractVector{<:Real}) = x
flat_params(x::AbstractArray{<:Real}) = vec(x)
flat_params(x::AbstractVector{<:AbstractVector{<:Real}}) = _flatten_vec_of_vec(x)
flat_params(x::AbstractVector{<:NTuple{N,<:Real}}) where N = _flatten_vec_of_vec(x)
flat_params(x::AbstractVector) = _flatten_vec_of_vec(flat_params.(x))
flat_params(A::DiagMatLike{<:Real}) = get_diagonal(A)
flat_params(A::ScalMat{<:Real}) = _svector((A.value,))
flat_params(A::PDMat{<:Real}) = flat_params(UpperTriangular(without_chol(A)))


function flat_params(A::UpperTriangular{<:Real})
    m = A.data
    # ToDo: Improve implementation
    reduce(vcat, [m[1:i, i] for i in 1:size(m, 1)])
end

function ChainRulesCore.rrule(::typeof(flat_params), A::UpperTriangular{<:Real})
    function _unshaped_pullback_uptri(thunked_x)
        x = unthunk(thunked_x)
        # ToDo: Improve implementation
        ΔA_data = zero(A.data)
        for j in axes(ΔA_data, 2)
            ΔA_data[1:j, j] = view(x, j*(j-1)÷2+1:j*(j+1)÷2)
        end
        return ChainRulesCore.NoTangent(), ProjectTo(A)(ΔA_data)
    end

    flat_params(A), _unshaped_pullback_uptri
end


# Since Distributions 0.25.133, `params` is a runtime fold over
# `namedparams`, which reverse-mode AD differentiates at great cost per
# call. The named parameters flatten to the same vector directly, so use
# them where a distribution provides them and `params` otherwise:
@static if isdefined(Distributions, :namedparams)
    _param_source(::Type{D}) where {D<:Distribution} =
        hasmethod(Distributions.namedparams, Tuple{D}) ? Val(:namedparams) : Val(:params)
else
    _param_source(::Type{<:Distribution}) = Val(:params)
end
ChainRulesCore.@non_differentiable _param_source(::Type)

_dist_params(d::Distribution) = _dist_params(_param_source(typeof(d)), d)
_dist_params(::Val{:namedparams}, d::Distribution) = values(Distributions.namedparams(d))
_dist_params(::Val{:params}, d::Distribution) = params(d)

function flat_params(d::Distribution)
    flat_params(_dist_params(d))
end

function flat_params(ntd::NamedTupleDist)
    flat_params(values(ntd))
end

# Flatten the components via their own `flat_params` methods, so that
# package extensions apply to them as well:
flat_params(dp::Distributions.Product) = flat_params(dp.v)
flat_params(dp::Distributions.ProductDistribution) = flat_params(vec(dp.dists))
