function get_nu_c(B)
    return EE * B/(2 * π * ME * CL)
end

function maxwell_juettner_I(B, θ, θe, ν, ne)
    νc = get_nu_c(B)

    νs = (2. /9.) * νc * θe * θ^2

    X = ν/νs

    prefactor = ne * EE^2 * νc/(CL)

    term1 = sqrt(2.) * π / 27. * sin(θ)
    term2 = ((X^(1. /2.)) + 2.0^(11. /12.) * X^(1. /6.))^2
    term3 = exp(-X^(1. /3.))

    ans = prefactor * term1 * term2 * term3

    if isnan(ans) || isinf(ans)
        println("Invalid maxwell_juettner_I calculation:")
        println("B = $B, θ = $θ, θe = $θe, ν = $ν, ne = $ne")
        println("Computed values: νc = $νc, νs = $νs, X = $X, prefactor = $prefactor, term1 = $term1, term2 = $term2, term3 = $term3")
        error("Resulting intensity is NaN or Inf")
    end
    return ans
end

