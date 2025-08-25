
using ForwardDiff

function Pos4ODE(Kcon::AbstractVector)
    return Kcon
end

function Mom4ODE(X::AbstractVector, Kcon::AbstractVector)
    lconn = get_connection_analytic(X)
    result = zeros(eltype(Kcon), 4)
    for mu in 1:4
        for alpha in 1:4
            for beta in 1:4
                result[mu] += lconn[mu, alpha, beta] * Kcon[alpha] * Kcon[beta]
            end
        end
    end
    return -result
end

function systemODEs_flat(XK)
    X = XK[1:4]
    Kcon = XK[5:8]
    return vcat(Pos4ODE(Kcon), Mom4ODE(X, Kcon))
end
function AutoDiffImageEulerMethod(ro, θo, spin, nx, ny, nmaxstep)
    fovx = 30/1000
    fovy = 30/1000

    #setting up X and Kcon initial conditions
    Xcam = camera_position(ro, θo, 0.0)
    X = MVec4(undef)
    Kcon = MVec4(undef)
    @inbounds for i in 1:nx
        @inbounds for j in 1:ny
            init_XK!(X, Kconi,j, Xcam, nx, fovx, fovy)
        end
    end


    Xarray = zeros(Float64, 4, nmaxstep)
    Kconarray = zeros(Float64, 4, nmaxstep)
    dX_dθo = zeros(Float64, 4, nmaxstep)
    dK_dθo = zeros(Float64, 4, nmaxstep)
    Xarray[:, 1] = Xcam
    Kconarray[:, 1] = Kcon
    step::Int64 = 2
    while (stop_backward_integration(X, Kcon) == 0)
        XK = vcat(X, Kcon)
        jac = ForwardDiff.jacobian(systemODEs_flat, XK)
        # Each column = input variable
        # jac is an 8×8 matrix:
        # Rows:    ODE output components (1:4 = dX₁/dλ, dX₂/dλ, dX₃/dλ, dX₄/dλ; 5:8 = dK₁/dλ, dK₂/dλ, dK₃/dλ, dK₄/dλ)
        # Columns: Input variables (1:4 = X₁, X₂, X₃, X₄; 5:8 = K₁, K₂, K₃, K₄)
        # Entry (i, j): ∂(ODE_i)/∂(var_j)
        # For example:
        #   jac[1,1] = ∂(dX₁/dλ)/∂X₁
        #   jac[1,5] = ∂(dX₁/dλ)/∂K₁
        #   jac[5,1] = ∂(dK₁/dλ)/∂X₁
        #   jac[8,8] = ∂(dK₄/dλ)/∂K₄
        # Table structure:
        #         | ∂(dX₁/dλ)/∂X₁ ... ∂(dX₁/dλ)/∂K₄ |
        #         | ∂(dX₂/dλ)/∂X₁ ... ∂(dX₂/dλ)/∂K₄ |
        #         | ∂(dX₃/dλ)/∂X₁ ... ∂(dX₃/dλ)/∂K₄ |
        #         | ∂(dX₄/dλ)/∂X₁ ... ∂(dX₄/dλ)/∂K₄ |
        #         | ∂(dK₁/dλ)/∂X₁ ... ∂(dK₁/dλ)/∂K₄ |
        #         | ∂(dK₂/dλ)/∂X₁ ... ∂(dK₂/dλ)/∂K₄ |
        #         | ∂(dK₃/dλ)/∂X₁ ... ∂(dK₃/dλ)/∂K₄ |
        #         | ∂(dK₄/dλ)/∂X₁ ... ∂(dK₄/dλ)/∂K₄ |
        println("jac size: ", size(jac, 1), "x", size(jac, 2))


        # Calculate the step size
        dl = stepsize(X, Kcon)


        dX_dθo[:, step] = dX_dθo[:, step  - 1] + dl * (jac[1:4, 1:4 ] * dX_dθo[:, step - 1] + jac[1:4, 5:8] * dK_dθo[:, step - 1])
        dK_dθo[:, step] = dK_dθo[:, step - 1] + dl * (jac[5:8, 1:4] * dX_dθo[:, step - 1] + jac[5:8, 5:8] * dK_dθo[:, step - 1])

        #Euler method
        Xarray[:, step] = Xarray[:, step - 1] + dl * Pos4ODE(Kcon)
        Kconarray[:, step] = Kconarray[:, step - 1] + dl * Mom4ODE(X, Kcon)
        step += 1
    
    end


end