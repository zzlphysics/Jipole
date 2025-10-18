function Xtoijk_ghost!(X, del)
    phi = rem(X[4], cstopx[4])
    if(phi < 0.0)
        phi += cstopx[4]
    end

    i = trunc(Int, ((X[2] - cstartx[2]) / dx[2]) - 0.5 + 1000) - 1000
    j = trunc(Int, ((X[3] - cstartx[3]) / dx[3]) - 0.5 + 1000) - 1000
    k = trunc(Int, ((phi - cstartx[4]) / dx[4]) - 0.5 + 1000) - 1000


    i = clamp(i, 0, N1 - 2)
    j = clamp(j, 0, N2 - 2)
    k = clamp(k, 0, N3 - 2)

    del[2] = (X[2] - ((i + 0.5) * dx[2] + cstartx[2])) / dx[2]
    del[3] = (X[3] - ((j + 0.5) * dx[3] + cstartx[3])) / dx[3]
    del[4] = (phi - ((k + 0.5) * dx[4] + cstartx[4])) / dx[4]

    del[2] = clamp(del[2], 0.0, 1.0)
    del[3] = clamp(del[3], 0.0, 1.0)
    del[4] = clamp(del[4], 0.0, 1.0)

    i += 1
    j += 1
    k += 1
    return i,j,k
end

function X_in_domain(X)
    if(X[2] < cstartx[2] || X[2] > cstopx[2] || X[3] < cstartx[3] || X[3] > cstopx[3])
        return 0
    end
    return 1
end

function ijktoX(i,j,k, X)
    X[2] = startx[2] + (i + 0.5) * dx[2]
    X[3] = startx[3] + (j + 0.5) * dx[3]
    X[4] = startx[4] + (k + 0.5) * dx[4]
    return
end


function interp_scalar(X, data)
    del::MVec4 = [0.0, 0.0, 0.0, 0.0]
    i,j,k = Xtoijk_ghost!(X, del)
    ip1 = i + 1
    jp1 = j + 1
    kp1 = k + 1

    b1 = 1. - del[2]
    b2 = 1. - del[3]
    #Interpolate in x1 and x2

    interp = data[i,j,k] * b1 * b2 +
             data[ip1,j,k] * del[2] * b2 +
             data[i,jp1,k] * b1 * del[3] +
             data[ip1,jp1,k] * del[2] * del[3]
    #Interpolate in x3
    interp = interp * (1. - del[4]) + (
             data[i,j,kp1] * b1 * b2 +
             data[ip1,j,kp1] * del[2] * b2 +
             data[i,jp1,kp1] * b1 * del[3] +
             data[ip1,jp1,kp1] * del[2] * del[3]) * del[4]
    return interp
end


function interp_scalar_time(X, dataA, dataB, tfac)
    vA = interp_scalar(X, dataA)
    if SLOW_LIGHT
        vB = interp_scalar(X, dataB)
        return (tfac) * vA + (1. -tfac) * vB
    end
    return vA
end



function gdet_zone(i,j,k)
    X::MVec4 = MVec4(undef)
    ijktoX(i,j,k, X)
    Xzone::MVec4 = MVec4(undef)
    Xzone[1] = 0.
    Xzone[2] = startx[2] + (i + 0.5) * dx[2]
    Xzone[3] = startx[3] + (j + 0.5) * dx[3]
    Xzone[4] = startx[4] + (k + 0.5) * dx[4]

    gcovKS::MMat4 = MMat4(undef)
    gcov::MMat4 = MMat4(undef)

    for μ in 1:NDIM
        for ν in 1:NDIM
            gcovKS[μ, ν] = 0.
            gcov[μ, ν] = 0.
        end
    end
    r,th = bl_coord(X)
    gcov_ks(r, th, bhspin, gcovKS)
    dxdX = set_dxdX(Xzone)

    for μ in 1:NDIM
        for ν in 1:NDIM
            for λ in 1:NDIM
                for κ in 1:NDIM
                    gcov[μ, ν] += dxdX[λ, ν] * dxdX[λ, μ] * gcovKS[λ, κ]
                end
            end
        end
    end

    return gdet_func(gcov)

end
