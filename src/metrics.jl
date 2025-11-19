using LinearAlgebra

function gdet_func(gcov)
    """
    Returns the determinant of the covariant metric tensor.

    Parameters:
    @gcov: Covariant metric tensor in Kerr-Schild coordinates.
    """
    F = lu(gcov)
    U = F.U

    if any(abs(U[i, i]) < 1e-14 for i in 1:size(U, 1))
        @warn "Singular matrix in gdet_func!"
        return -1.0
    end

    gdet = prod(diag(U))
    return sqrt(abs(gdet))
end

function gcov_func!(X, bhspin, gcov,R0::Float64 = 0.0)
    """
    Returns covariant metric tensor in Kerr-Schild coordinates.

    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    r, th = bl_coord(X)
    cth = cos(th)
    sth = abs(sin(th))

    s2 = sth * sth
    rho2 = r * r + bhspin * bhspin * cth * cth

    tfac = 1.
    rfac = r - R0
    hfac = π
    pfac = 1.
    gcov[1, 1] = (-1. + 2. * r / rho2) * tfac * tfac
    gcov[1, 2] = (2. * r / rho2) * tfac * rfac
    gcov[1, 4] = (-2. * bhspin * r * s2 / rho2) * tfac * pfac

    gcov[2, 1] = gcov[1, 2]
    gcov[2, 2] = (1. + 2. * r / rho2) * rfac * rfac
    gcov[2, 4] = (-bhspin * s2 * (1. + 2. * r / rho2)) * rfac * pfac
    
    gcov[3, 3] = rho2 * hfac * hfac
    
    gcov[4, 1] = gcov[1, 4]
    gcov[4, 2] = gcov[2, 4]
    gcov[4, 4] =
        s2 * (rho2 + bhspin * bhspin * s2 * (1. + 2. * r / rho2)) * pfac * pfac



    # Assert if the diagonal elements are zero
    if gcov[1, 1] == 0 || gcov[2, 2] == 0 || gcov[3, 3] == 0 || gcov[4, 4] == 0
        @error "Singular gcov encountered in gcov_func"
        println("sth $sth, cth $cth, r $r, a $bhspin, rho2 $rho2, tfac $tfac, rfac $rfac, hfac $hfac, pfac $pfac")
        println("X = $X")
        println("th = $th")
        print_matrix("gcov", gcov)
        error("Singular gcov encountered, cannot compute gcov_func.")
    end
end

function gcov_func(X, bhspin, R0::Float64 = 0.0)
    """
    Returns covariant metric tensor in Kerr-Schild coordinates.

    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    r, th = bl_coord(X)
    T = promote_type(typeof(r), typeof(th), typeof(bhspin))
    gcov = @MMatrix zeros(T, 4, 4)
    cth = cos(th)
    sth = abs(sin(th))

    s2 = sth * sth
    rho2 = r * r + bhspin * bhspin * cth * cth

    tfac = 1.
    rfac = r - R0
    if(MODEL == "iharm")
        if(METRIC == "MKS")
            hfac = π * (1.0 + (1.0 - hslope) * cos(2.0 * π * X[3]))
            dth_dX2 = 0.0
        elseif(METRIC == "FMKS")
            # FMKS metric
            E = exp(mks_smooth * (startx[2] - X[2]))
            dthG = π * (1.0 + (1.0 - hslope) * cos(2.0 * π * X[3]))
            y = 2 * X[3] - 1.0
            dthJ = 2 * poly_norm * (1 + (y/poly_xt)^poly_alpha)
            dthG2 = -2 * π * π * (1.0 - hslope) * sin(2.0 * π * X[3])
            dthJ2 = 4 * poly_norm * poly_alpha * (y/poly_xt)^(poly_alpha - 1) / poly_xt
            hfac = (1.0 - E) * dthG + E * dthJ
            thG = π * X[3] + ((1. - hslope) / 2.) * sin(2. * π * X[3]);
            thJ = poly_norm * y* (1. + ((y / poly_xt)^poly_alpha) / (poly_alpha + 1.)) + 0.5 * π;
            dth_dX2 = -mks_smooth * exp(mks_smooth * (startx[2] - X[2])) * (thJ - thG)
        else
            error("Unknown METRIC type: $METRIC")
        end
    elseif(MODEL == "analytic" || MODEL == "thin_disk")
        hfac = π
        dth_dX2 = 0.0
    end
    pfac = 1.
    gcov[1, 1] = (-1. + 2. * r / rho2) * tfac * tfac
    gcov[1, 2] = (2. * r / rho2) * tfac * rfac
    gcov[1, 4] = (-2. * bhspin * r * s2 / rho2) * tfac * pfac

    gcov[2, 1] = gcov[1, 2]
    gcov[2, 2] = (1. + 2. * r / rho2) * rfac * rfac
    gcov[2, 3] = rho2 * dth_dX2 * hfac
    gcov[2, 4] = (-bhspin * s2 * (1. + 2. * r / rho2)) * rfac * pfac
    
    gcov[3,2] = gcov[2,3]
    gcov[3, 3] = rho2 * hfac * hfac
    
    gcov[4, 1] = gcov[1, 4]
    gcov[4, 2] = gcov[2, 4]
    gcov[4, 4] =
        s2 * (rho2 + bhspin * bhspin * s2 * (1. + 2. * r / rho2)) * pfac * pfac



    # Assert if the diagonal elements are zero
    if gcov[1, 1] == 0 || gcov[2, 2] == 0 || gcov[3, 3] == 0 || gcov[4, 4] == 0
        @error "Singular gcov encountered in gcov_func"
        println("sth $sth, cth $cth, r $r, a $bhspin, rho2 $rho2, tfac $tfac, rfac $rfac, hfac $hfac, pfac $pfac")
        println("X = $X")
        println("th = $th")
        print_matrix("gcov", gcov)
        error("Singular gcov encountered, cannot compute gcov_func.")
    end
    
        return gcov
end



function gcov_func_fd(X, bhspin, R0::Float64 = 0.0)
    """
    Returns covariant metric tensor in Kerr-Schild coordinates.

    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    r, th = bl_coord(X)
    T = promote_type(typeof(r), typeof(th), typeof(bhspin))
    gcov = @MMatrix zeros(T, 4, 4)
    Gcov_ks = @MMatrix zeros(T, 4, 4)
    gcov_ks(r, th, bhspin, Gcov_ks)



    dxdX = set_dxdX(X)
    for μ in 1:NDIM
        for ν in 1:NDIM
            for λ in 1:NDIM
                for κ in 1:NDIM
                    gcov[μ, ν] +=  Gcov_ks[λ, κ] * dxdX[λ, μ] * dxdX[κ, ν] 
                end
            end
        end
    end
    return gcov
end

function gcon_func!(gcov, gcon)
    """
    Returns contravariant metric tensor in Kerr-Schild coordinates through matrix inversion of the covariant tensor.
    Parameters:
    @gcov: Covariant metric tensor in Kerr-Schild coordinates.
    """
    gcon .= inv(gcov)
    if any(isnan.(gcon)) || any(isinf.(gcon))
        @error "Singular gcov encountered in gcon"
        print_matrix("gcov", gcov)
        print_matrix("gcon", gcon)
        error("Singular gcov encountered, cannot compute gcon.")
    end
end

function gcon_func(gcov)
    """
    Returns contravariant metric tensor in Kerr-Schild coordinates through matrix inversion of the covariant tensor.
    Parameters:
    @gcov: Covariant metric tensor in Kerr-Schild coordinates.
    """
    gcon = inv(gcov)
    if any(isnan.(gcon)) || any(isinf.(gcon))
        @error "Singular gcov encountered in gcon"
        print_matrix("gcov", gcov)
        print_matrix("gcon", gcon)
        error("Singular gcov encountered, cannot compute gcon.")
    end
    return gcon
end

function gcov_bl!(r,th, bhspin, gcov)
    """
    Computes the metric tensor in Boyer-Lindquist coordinates.
    Parameters:
    @r: Radial coordinate in Boyer-Lindquist coordinates.
    @th: Angular coordinate in Boyer-Lindquist coordinates.
    """

    sth = sin(th)
    if(sth < 1e-40)
        sth = 10^(-40)
    end
    cth = cos(th)
    s2 = sth * sth
    if(r < 1e-40)
        r = 10^(-40)
    end
    a2 = bhspin * bhspin
    r2 = r * r
    DD = (1.0 - 2.0 / r + a2 / r2)
    mu = 1.0 + a2 * cth * cth / r2

    gcov[1, 1] = -(1.0 - 2.0 / (r * mu))
    gcov[1, 4] = -2.0 * bhspin * s2 / (r * mu)
    gcov[4, 1] = gcov[1, 4]
    gcov[2, 2] = mu / (DD )
    gcov[3, 3] = r2 * mu
    gcov[4, 4] = r2 * sth * sth * (1.0 + a2 / r2 + 2.0 * a2 * s2 / (r2 * r * mu))

    #if any element of the diagonal is zero print variables
    if(gcov[1,1] == 0 || gcov[2,2] == 0 || gcov[3,3] == 0 || gcov[4,4] == 0)   
        @error "Singular gcov encountered in gcov_bl"
        println("sth $sth, cth $cth, r $r, a $bhspin, r2 $r2, a2 $a2, mu $mu, DD $DD")
        print_matrix("gcov", gcov)
        error("Singular gcov encountered, cannot compute gcov_bl.")
    end
    #if any (isnan.(gcov)) || any(isinf.(gcov))
    if any(isnan.(gcov)) || any(isinf.(gcov))
        @error "Singular gcov encountered in gcov_bl"
        println("sth $sth, cth $cth, r $r, a $bhspin, r2 $r2, a2 $a2, mu $mu, DD $DD")
        print_matrix("gcov", gcov)
        error("Singular gcov encountered, cannot compute gcov_bl.")
    end
end