function get_fluid_nu(Kcon, Ucov)
    """
    Computes the fluid frequency from the covariant 4-vector and the 4-velocity.
    Parameters:
    @Kcon: Covariant 4-vector of the photon in internal coordinates.
    @Ucov: Covariant 4-velocity of the fluid in internal coordinates.
    """
    nu = - (Kcon[1] * Ucov[1] + Kcon[2] * Ucov[2] + Kcon[3] * Ucov[3] + Kcon[4] * Ucov[4]) * ME * CL * CL / HPL
    return nu
end

function get_jk(X, Kcon, freq::Float64, bhspin)

    if MODEL == "analytic"
        return get_analytic_jk(X, Kcon, freq, bhspin)
    elseif MODEL == "thin_disk"
        return (zero(eltype(X)), zero(eltype(X)))
    else
        error("Unknown model: $MODEL")
    end
end


function integrate_emission!(traj::Vector{OfTraj}, nsteps::Int, Image::Matrix{Float64}, I::Int, J::Int, freq::Float64, bhspin::Float64)
    """
    Integrates the emission along the geodesic trajectory.
    
    Parameters:
    @traj: Vector of OfTraj objects containing the geodesic trajectory.
    @nstep: Number of steps in the trajectory.
    @Image: Matrix to store the integrated intensity.
    @I: x-index of the pixel in the image plane.
    @J: y-index of the pixel in the image plane.

    """
    Xi = MVec4(undef)
    Kconi = MVec4(undef)
    Xf = MVec4(undef)
    Kconf = MVec4(undef)
    Rh::Float64 = 1 + sqrt(1. - bhspin * bhspin);
    #Take the value of the initial X component
     for k in 1:NDIM
            Xi[k] = traj[nsteps].X[k]
            Kconi[k] = traj[nsteps].Kcon[k]
    end

    ji, ki = get_jk(Xi, Kconi, freq, bhspin)
    if(I == 152 && J == 171)
        println("Initial intensity at pixel (151, 170) = $ji")
        r, th = bl_coord(Xi)
        println("Initial position: r = $r, th = $th")
        println("Initial coefficients: ji = $ji, ki = $ki")
        println("nsteps = $nsteps")
        println("Xi = $Xi")
        println("Kconi = $Kconi")
    end

    Intensity = 0.0
    for nstep = nsteps:-1:2
        for k in 1:NDIM
            Xi[k] = traj[nstep].X[k]
            Xf[k] = traj[nstep - 1].X[k]
            Kconi[k] = traj[nstep].Kcon[k]
            Kconf[k] = traj[nstep - 1].Kcon[k]
        end
        if(MODEL == "thin_disk")
            if(thindisk_region(Xi, Xf))
                Intensity = GetTDBoundaryCondition(Xi, Kconi, bhspin, Rh)
            end
            continue
        end

        if !radiating_region(Xf, Rh)
            nstep -= 1
            continue
        end


        jf, kf = get_jk(Xf, Kconf, freq, bhspin)
        Intensity::Float64 = approximate_solve(Intensity, ji, ki, jf, kf, traj[nstep - 1].dl)
        # if (I == 152 && J == 171)
        #     println("Intensity at pixel (151, 170) after step $nstep = $Intensity")
        #     println("ji = $ji, ki = $ki, jf = $jf, kf = $kf, dl = $(traj[nstep - 1].dl)")
        # end

        if (isnan(Intensity) || isinf(Intensity))
            @error "NaN or Inf encountered in intensity calculation at pixel ($I, $J)"
            println("Intensity = $Intensity")
            print_vector("Kconf =", Kconf)
            print_vector("Kconi =", Kconi)

            error("NaN or Inf encountered in intensity calculation")
        end
        ji = jf
        ki = kf
        nstep -= 1
    end
    Image[I, J] = Intensity
end

function get_bk_angle(Kcon::MVec4, Ucov::MVec4, Bcon::MVec4, Bcov::MVec4)
    """
    Computes the angle between the photon 4-momentum and the magnetic field 4-vector.
    
    Parameters:
    @Kcon: Covariant 4-vector of the photon in internal coordinates.
    @Ucov: Covariant 4-velocity of the fluid in internal coordinates.
    @Bcon: Covariant 4-vector of the magnetic field in internal coordinates.
    @Bcov: Covariant 4-vector of the magnetic field in internal coordinates.
    
    Returns:
    The angle between the photon momentum and the magnetic field.
    """
    # Calculating the module of the magnetic field 4-vector
    norm_B = sqrt(abs(Bcov[1]*Bcon[1] + Bcov[2]*Bcon[2] + Bcov[3]*Bcon[3] + Bcov[4]*Bcon[4]))
    
    if(norm_B == 0)
        return π/2
    end
    # Calculating the module of the photon 4-momentum
    norm_K = abs(Kcon[1]*Ucov[1] + Kcon[2]*Ucov[2] + Kcon[3]*Ucov[3] + Kcon[4]*Ucov[4])
    
    #Calculating the normalized dot product of the photon 4-momentum and the magnetic field 4-vector
    μ = (Kcon[1] * Bcov[1] + Kcon[2] * Bcov[2] + Kcon[3] * Bcov[3] + Kcon[4] * Bcov[4]) / (norm_K * norm_B)

    if abs(μ) > 1.
        μ /= abs(μ)
    end
    
    return acos(μ)
end

function approximate_solve(Ii, ji, ki, jf, kf, dl)
    """
    Evolves the intensity along the geodesic using an approximate method.

    Parameters:
    @Ii: Initial intensity at the start of the segment.
    @ji: Emissivity at the start of the segment.
    @ki: Absorption coefficient at the start of the segment.
    @jf: Emissivity at the end of the segment.
    @kf: Absorption coefficient at the end of the segment.
    @dl: Length of the segment along the geodesic.
    """


    If = 0.0
    javg = (ji + jf) / 2.
    kavg = (ki + kf) / 2.

    dtau = dl * kavg

    if (dtau < 1.e-3)
    If = Ii + (javg - Ii * kavg) * dl * (1. - (dtau / 2.) * (1. - dtau / 3.))
    else
    efac = exp(-dtau)
    If = Ii * efac + (javg / kavg) * (1. - efac)
    end

    if(isnan(If) || isinf(If))
        @error "Invalid intensity computed" If
        println("Ii = $Ii, ji = $ji, ki = $ki, jf = $jf, kf = $kf, dl = $dl")
        println("dtau = $dtau, javg = $javg, kavg = $kavg")
        println("efac = $(exp(-dtau))")
        error("Invalid intensity computed: $If")
    end

    return If
end
