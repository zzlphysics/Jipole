function Xtoijk_ghost!(X, del)
    # --- i (r) and j (theta) ---
    
    i_logical = trunc(Int, ((X[2] - startx[2]) / dx[2]) - 0.5 + 1000) - 1000
    j_logical = trunc(Int, ((X[3] - startx[3]) / dx[3]) - 0.5 + 1000) - 1000

    i = clamp(i_logical, 0, N1 - 2) # Clamps logical r-index
    j = clamp(j_logical, 0, N2 - 2) # Clamps logical theta-index

    del[2] = (X[2] - ((i + 0.5) * dx[2] + startx[2])) / dx[2]
    del[3] = (X[3] - ((j + 0.5) * dx[3] + startx[3])) / dx[3]

    del[2] = clamp(del[2], 0.0, 1.0)
    del[3] = clamp(del[3], 0.0, 1.0)

    i += 1 # Final 1-based index for i
    j += 1 # Final 1-based index for j

    # --- k (phi) PERIODIC FIX ---
    phi = rem(X[4], cstopx[4])
    if(phi < 0.0)
        phi += cstopx[4]
    end

    # Calculate the logical k-index. This can range from -1 to N3
    k_logical = trunc(Int, ((phi - startx[4]) / dx[4]) - 0.5 + 1000) - 1000
    
    # Calculate the fractional part (delta) using the *unclamped* logical index
    del[4] = (phi - ((k_logical + 0.5) * dx[4] + startx[4])) / dx[4]
    del[4] = clamp(del[4], 0.0, 1.0)

    # Wrap the logical k-index to the valid 0-based range [0, N3-1]
    k_logical_wrapped = mod(k_logical, N3) 
    
    # Convert to 1-based Julia index [1, N3]
    k = k_logical_wrapped + 1

    return i, j, k
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



# function interp_scalar(X, data)
#     # del corresponds to offsets: del[2]->i, del[3]->j, del[4]->k
#     del = zeros(eltype(X), 4)

#     # Get base indices (bottom-left of the 2x2x2 cube in trilinear)
#     i_base, j_base, k_base = Xtoijk_ghost!(X, del)

#     (N1, N2, N3) = size(data)

#     # --- 1. Compute Cubic Weights ---
#     # We use Catmull-Rom spline weights. 
#     # These return a 4-element tuple/vector of weights for neighbors -1, 0, +1, +2
#     @inline function get_cubic_weights(t)
#         t2 = t * t
#         t3 = t2 * t
        
#         # Weights for p[-1], p[0], p[1], p[2]
#         w0 = -0.5 * t3 +       t2 - 0.5 * t
#         w1 =  1.5 * t3 - 2.5 * t2 + 1.0
#         w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
#         w3 =  0.5 * t3 - 0.5 * t2
#         return (w0, w1, w2, w3)
#     end

#     w_i = get_cubic_weights(del[2])
#     w_j = get_cubic_weights(del[3])
#     w_k = get_cubic_weights(del[4])

#     # --- 2. Interpolation Loop ---
#     val = zero(eltype(data))

#     # We loop over the 4x4x4 neighborhood
#     # m, n, p are offsets relative to base indices: -1, 0, 1, 2
#     # But since Julia is 1-based and our weights tuple is 1-based:
#     # weight index 1 -> offset -1
#     # weight index 2 -> offset  0 (the base index)
#     # weight index 3 -> offset +1
#     # weight index 4 -> offset +2

#     for (idx_k, offset_k) in enumerate(-1:2)
        
#         # --- K PERIODIC FIX (Extended) ---
#         k_curr = k_base + offset_k
#         if k_curr < 1
#             k_curr = k_curr + N3
#         elseif k_curr > N3
#             k_curr = k_curr - N3
#         end
#         # ---------------------------------

#         wk = w_k[idx_k]

#         for (idx_j, offset_j) in enumerate(-1:2)
            
#             # --- J Boundary (Clamp) ---
#             j_curr = clamp(j_base + offset_j, 1, N2)
#             wj = w_j[idx_j]

#             for (idx_i, offset_i) in enumerate(-1:2)
                
#                 # --- I Boundary (Clamp) ---
#                 i_curr = clamp(i_base + offset_i, 1, N1)
#                 wi = w_i[idx_i]

#                 # Accumulate weighted sum
#                 val += data[i_curr, j_curr, k_curr] * wi * wj * wk
#             end
#         end
#     end
#     # --- 3. SAFETY CLAMP FOR PHYSICS ---
#     # Cubic splines can "overshoot" into negative values near sharp gradients 
#     # (like a black hole horizon or jet edge). 
#     # Since Density/Temp must be > 0, we clamp the bottom to a tiny number.
#     return max(val, 1e-2)
#     return val
# end

# function interp_scalar(X, data)
#     #del::MVec4 = [0.0, 0.0, 0.0, 0.0]
#     del = zeros(eltype(X), 4)

#     i, j, k = Xtoijk_ghost!(X, del)

#     (N1_data, N2_data, N3_data) = size(data) 

#     ip1 = i + 1
#     jp1 = j + 1

#     # --- k (phi) PERIODIC FIX ---
#     # k is the base index (e.g., 1 to 32)
#     kp1 = k + 1
#     if kp1 > N3_data
#         kp1 = 1  # Wrap around to the first cell
#     end
#     # --- END FIX ---

#     b1 = 1.0 - del[2]
#     b2 = 1.0 - del[3]

#     # Interpolate in i and j
#     interp = data[i, j, k]   * b1 * b2 +
#              data[ip1, j, k] * del[2] * b2 +
#              data[i, jp1, k] * b1 * del[3] +
#              data[ip1, jp1, k] * del[2] * del[3]

#     # Interpolate in k (phi)
#     # This uses the wrapped kp1, so it correctly interpolates
#     # between the last cell (k=32) and the first cell (kp1=1).
#     interp = interp * (1.0 - del[4]) + (
#              data[i, j, kp1]   * b1 * b2 +
#              data[ip1, j, kp1] * del[2] * b2 +
#              data[i, jp1, kp1] * b1 * del[3] +
#              data[ip1, jp1, kp1] * del[2] * del[3]
#     ) * del[4]

#     return interp
# end

function interp_scalar(X, data)
    del = zeros(eltype(X), 4)
    i,j,k = Xtoijk_ghost!(X, del)

    if(data[i,j,k] <= 1.)
        return 1.
    end

    return data[i,j,k]
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
    rt = MVector{2,Float64}(undef)
    bl_coord!(rt, X)
    r = rt[1]
    th = rt[2]
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
