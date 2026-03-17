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



function gcov_func!(X::MVec4, bhspin::Float64, gcov, R0::Float64 = 0.0)
    """
    Returns g_{munu} at location specified by X.
    Adapted from ipole C code logic.
    """
    
    # Get Boyer-Lindquist coordinates (r, theta)
    # Assumes bl_coord is defined elsewhere
    r, th = bl_coord(X) 
    
    # Initialize metric to zero
    fill!(gcov, 0.0)

    # --- Case 1: Minkowski (Spherical Polar) ---
    if params.metric == METRIC_MINKOWSKI
        gcov[1, 1] = -1.0
        gcov[2, 2] = 1.0
        gcov[3, 3] = r * r
        gcov[4, 4] = r * r * sin(th)^2
        return

    # --- Case 2: E-Minkowski (Exponential Radial, Spherical Polar) ---
    elseif params.metric == METRIC_EMINKOWSKI
        # In E-Minkowski, r = exp(X[2]), so dr/dX[2] = r
        gcov[1, 1] = -1.0
        gcov[2, 2] = r * r  # g_rr * (dr/dX)^2 = 1 * r^2
        gcov[3, 3] = r * r
        gcov[4, 4] = r * r * sin(th)^2
        return

    # --- Case 3: FMKS (Funky Modified Kerr-Schild) - Analytic ---
    elseif params.metric == METRIC_FMKS
        sth = sin(th)
        cth = cos(th)
        s2 = sth^2
        rho2 = r^2 + bhspin^2 * cth^2

        # 1. Compute Standard KS components (cyclic in t, phi)
        # Note: These are temporary values for calculation, but we assign 
        # directly where they align or are part of the transform.
        
        # g_tt_ks = -1 + 2r/rho2
        # g_tr_ks = 2r/rho2
        # g_tphi_ks = -2*a*r*s2/rho2
        
        # In FMKS, X[1]=t, X[4]=phi are unchanged, so g_tt and g_tphi maps directly
        gcov[1, 1] = -1.0 + 2.0 * r / rho2
        gcov[1, 4] = -2.0 * bhspin * r * s2 / rho2
        gcov[4, 1] = gcov[1, 4]
        
        # g_phiphi maps directly
        gcov[4, 4] = s2 * (rho2 + bhspin^2 * s2 * (1.0 + 2.0 * r / rho2))

        # 2. Compute Coordinate Transformation derivatives for r(x2) and th(x3)
        # Note: Julia X[2] corresponds to C X[1] (radius)
        #       Julia X[3] corresponds to C X[2] (theta)
        
        E = exp(params.mks_smooth * (params.startx[2] - X[2]))
        
        # Theta transformation parts
        dthG = π * (1.0 + (1.0 - params.hslope) * cos(2.0 * π * X[3]))
        y = 2.0 * X[3] - 1.0
        dthJ = 2.0 * params.poly_norm * (1.0 + (y / params.poly_xt)^params.poly_alpha)
        
        # Hfac is d(theta)/d(X3)
        hfac = (1.0 - E) * dthG + E * dthJ
        
        # d(theta)/d(X2) - The "Funky" part
        thG = π * X[3] + ((1.0 - params.hslope) / 2.0) * sin(2.0 * π * X[3])
        thJ = params.poly_norm * y * (1.0 + ((y / params.poly_xt)^params.poly_alpha) / (params.poly_alpha + 1.0)) + 0.5 * π
        dth_dX2 = -params.mks_smooth * E * (thJ - thG)

        # 3. Apply FMKS Transformations to remaining components
        
        # g_tx (mixed term) = g_tr_ks * dr/dX2 = (2r/rho2) * r
        # Since r = exp(x2), dr/dX2 = r
        gcov[1, 2] = (2.0 * r / rho2) * r
        gcov[2, 1] = gcov[1, 2]

        # g_xx (radial) = g_rr_ks * (dr/dX2)^2
        # g_rr_ks = 1 + 2r/rho2
        gcov[2, 2] = (1.0 + 2.0 * r / rho2) * r * r

        # g_xy (radial-theta cross term) = g_rr_ks * dr/dx * dr/dy + g_thth_ks * dth/dx * dth/dy
        # ... Wait, actually it simplifies to: rho2 * dth/dX2 * dth/dX3
        # Because g_r_theta_ks is 0. 
        gcov[2, 3] = rho2 * dth_dX2 * hfac
        gcov[3, 2] = gcov[2, 3]

        # g_yy (theta) = g_thth_ks * (dth/dX3)^2
        gcov[3, 3] = rho2 * hfac * hfac

        # g_xphi = g_rphi_ks * dr/dX2
        gcov[2, 4] = -bhspin * s2 * (1.0 + 2.0 * r / rho2) * r
        gcov[4, 2] = gcov[2, 4]

        return
    end

    # --- Case 4: Generic / MKS (Matrix Multiplication) ---
    # We calculate KS metric, then transform it using Jacobian dxdX.
    
    # 1. Compute KS Metric (Gcov_ks)
    # We use a temporary MMatrix or regular array
    Gcov_ks = @MMatrix zeros(4, 4)
    
    cth = cos(th)
    sth = sin(th)
    s2 = sth^2
    rho2 = r^2 + bhspin^2 * cth^2
    
    Gcov_ks[1, 1] = -1.0 + 2.0 * r / rho2
    Gcov_ks[1, 2] = 2.0 * r / rho2
    Gcov_ks[1, 4] = -2.0 * bhspin * r * s2 / rho2
    
    Gcov_ks[2, 1] = Gcov_ks[1, 2]
    Gcov_ks[2, 2] = 1.0 + 2.0 * r / rho2
    Gcov_ks[2, 4] = -bhspin * s2 * (1.0 + 2.0 * r / rho2)
    
    Gcov_ks[3, 3] = rho2
    
    Gcov_ks[4, 1] = Gcov_ks[1, 4]
    Gcov_ks[4, 2] = Gcov_ks[2, 4]
    Gcov_ks[4, 4] = s2 * (rho2 + bhspin^2 * s2 * (1.0 + 2.0 * r / rho2))

    # 2. Get Jacobian dxdX (Transformation Matrix)
    # dxdX[mu, nu] = d(KS_mu) / d(Internal_nu)
    dxdX = set_dxdX(X)

    # 3. Matrix Multiplication: gcov = J^T * G_ks * J
    # gcov[mu][nu] += Gcov_ks[lam][kap] * dxdX[lam][mu] * dxdX[kap][nu]
    
    fill!(gcov, 0.0)
    for mu in 1:4
        for nu in 1:4
            sum_val = 0.0
            for lam in 1:4
                for kap in 1:4
                    sum_val += Gcov_ks[lam, kap] * dxdX[lam, mu] * dxdX[kap, nu]
                end
            end
            gcov[mu, nu] = sum_val
        end
    end
end

function gcov_func(X, bhspin, R0::Float64 = 0.0)
    """
    Returns g_{munu} at location specified by X.
    Adapted from ipole C code logic.
    """
    
    # Get Boyer-Lindquist coordinates (r, theta)
    # Assumes bl_coord is defined elsewhere
    r, th = bl_coord(X) 
    T = promote_type(typeof(r), typeof(th), typeof(bhspin))
    gcov = @MMatrix zeros(T, 4, 4)

    # Initialize metric to zero
    fill!(gcov, 0.0)
    if(MODEL == "iharm")
        # --- Case 1: Minkowski (Spherical Polar) ---
        if params.metric == METRIC_MINKOWSKI
            gcov[1, 1] = -1.0
            gcov[2, 2] = 1.0
            gcov[3, 3] = r * r
            gcov[4, 4] = r * r * sin(th)^2
            return gcov

        # --- Case 2: E-Minkowski (Exponential Radial, Spherical Polar) ---
        elseif params.metric == METRIC_EMINKOWSKI
            # In E-Minkowski, r = exp(X[2]), so dr/dX[2] = r
            gcov[1, 1] = -1.0
            gcov[2, 2] = r * r  # g_rr * (dr/dX)^2 = 1 * r^2
            gcov[3, 3] = r * r
            gcov[4, 4] = r * r * sin(th)^2
            return gcov

        # --- Case 3: FMKS (Funky Modified Kerr-Schild) - Analytic ---
        elseif params.metric == METRIC_FMKS
            cth = cos(th)
            sth = sin(th)
            s2 = sth^2
            rho2 = r^2 + bhspin^2 * cth^2

            # 1. Compute Standard KS components (cyclic in t, phi)
            # Note: These are temporary values for calculation, but we assign 
            # directly where they align or are part of the transform.
            
            # g_tt_ks = -1 + 2r/rho2
            # g_tr_ks = 2r/rho2
            # g_tphi_ks = -2*a*r*s2/rho2
            
            # In FMKS, X[1]=t, X[4]=phi are unchanged, so g_tt and g_tphi maps directly
            gcov[1, 1] = -1.0 + 2.0 * r / rho2
            gcov[1, 4] = -2.0 * bhspin * r * s2 / rho2
            gcov[4, 1] = gcov[1, 4]
            
            # g_phiphi maps directly
            gcov[4, 4] = s2 * (rho2 + bhspin^2 * s2 * (1.0 + 2.0 * r / rho2))

            # 2. Compute Coordinate Transformation derivatives for r(x2) and th(x3)
            # Note: Julia X[2] corresponds to C X[1] (radius)
            #       Julia X[3] corresponds to C X[2] (theta)
            
            E = exp(params.mks_smooth * (params.startx[2] - X[2]))
            
            # Theta transformation parts
            dthG = π * (1.0 + (1.0 - params.hslope) * cos(2.0 * π * X[3]))
            y = 2.0 * X[3] - 1.0
            dthJ = 2.0 * params.poly_norm * (1.0 + (y / params.poly_xt)^params.poly_alpha)
            
            # Hfac is d(theta)/d(X3)
            hfac = (1.0 - E) * dthG + E * dthJ
            
            # d(theta)/d(X2) - The "Funky" part
            thG = π * X[3] + ((1.0 - params.hslope) / 2.0) * sin(2.0 * π * X[3])
            thJ = params.poly_norm * y * (1.0 + ((y / params.poly_xt)^params.poly_alpha) / (params.poly_alpha + 1.0)) + 0.5 * π
            dth_dX2 = -params.mks_smooth * E * (thJ - thG)

            # 3. Apply FMKS Transformations to remaining components
            
            # g_tx (mixed term) = g_tr_ks * dr/dX2 = (2r/rho2) * r
            # Since r = exp(x2), dr/dX2 = r
            gcov[1, 2] = (2.0 * r / rho2) * r
            gcov[2, 1] = gcov[1, 2]

            # g_xx (radial) = g_rr_ks * (dr/dX2)^2
            # g_rr_ks = 1 + 2r/rho2
            gcov[2, 2] = (1.0 + 2.0 * r / rho2) * r * r

            # g_xy (radial-theta cross term) = g_rr_ks * dr/dx * dr/dy + g_thth_ks * dth/dx * dth/dy
            # ... Wait, actually it simplifies to: rho2 * dth/dX2 * dth/dX3
            # Because g_r_theta_ks is 0. 
            gcov[2, 3] = rho2 * dth_dX2 * hfac
            gcov[3, 2] = gcov[2, 3]

            # g_yy (theta) = g_thth_ks * (dth/dX3)^2
            gcov[3, 3] = rho2 * hfac * hfac

            # g_xphi = g_rphi_ks * dr/dX2
            gcov[2, 4] = -bhspin * s2 * (1.0 + 2.0 * r / rho2) * r
            gcov[4, 2] = gcov[2, 4]

            return gcov
        end

        # --- Case 4: Generic / MKS (Matrix Multiplication) ---
        # We calculate KS metric, then transform it using Jacobian dxdX.
        
        # 1. Compute KS Metric (Gcov_ks)
        Gcov_ks = similar(gcov)
        fill!(Gcov_ks, 0.0)
        
        cth = cos(th)
        sth = sin(th)
        s2 = sth^2
        rho2 = r^2 + bhspin^2 * cth^2
        Gcov_ks[1, 1] = -1.0 + 2.0 * r / rho2
        Gcov_ks[1, 2] = 2.0 * r / rho2
        Gcov_ks[1, 4] = -2.0 * bhspin * r * s2 / rho2
        
        Gcov_ks[2, 1] = Gcov_ks[1, 2]
        Gcov_ks[2, 2] = 1.0 + 2.0 * r / rho2
        Gcov_ks[2, 4] = -bhspin * s2 * (1.0 + 2.0 * r / rho2)
        
        Gcov_ks[3, 3] = rho2
        
        Gcov_ks[4, 1] = Gcov_ks[1, 4]
        Gcov_ks[4, 2] = Gcov_ks[2, 4]
        Gcov_ks[4, 4] = s2 * (rho2 + bhspin^2 * s2 * (1.0 + 2.0 * r / rho2))

        # 2. Get Jacobian dxdX (Transformation Matrix)
        # dxdX[mu, nu] = d(KS_mu) / d(Internal_nu)
        dxdX = set_dxdX(X)
        # 3. Matrix Multiplication: gcov = J^T * G_ks * J
        # gcov[mu][nu] += Gcov_ks[lam][kap] * dxdX[lam][mu] * dxdX[kap][nu]
        
        fill!(gcov, 0.0)
        for mu in 1:4
            for nu in 1:4
                sum_val = 0.0
                for lam in 1:4
                    for kap in 1:4
                        sum_val += Gcov_ks[lam, kap] * dxdX[lam, mu] * dxdX[kap, nu]
                    end
                end
                gcov[mu, nu] = sum_val
            end
        end
        return gcov
    else
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

        return gcov
    end
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