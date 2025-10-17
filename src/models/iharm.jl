module iharm

using HDF5
using Printf

# --- Public Interface ---
export IharmData, load_data, get_p, view_slice
export RHO, UU, U1, U2, U3, B1, B2, B3

# --- Constants for Primitives ---
const VALID_PRIMS = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3"]
const RHO = "RHO"; const UU  = "UU"; const U1  = "U1"; const U2  = "U2";
const U3  = "U3";  const B1  = "B1"; const B2  = "B2"; const B3  = "B3";


# --- Data Storage Structure ---
"""
    IharmData

A struct to hold the data from an iharm3d simulation dump.
The data is stored in a dictionary where keys are the primitive names
(e.g., "RHO") and values are 3D arrays.
"""
struct IharmData
    primitives::Dict{String, Array{Float64, 3}}
end


# --- Internal Functions ---

function _read_single_primitive(file_handle, prim_name::String)
    if "prims" in keys(file_handle)
        prims = read(file_handle["prims"])
        prim_idx = findfirst(isequal(uppercase(prim_name)), VALID_PRIMS)
        # Returns 'nothing' if the primitive is not found
        return prim_idx === nothing ? nothing : permutedims(prims[prim_idx, :, :, :], (3, 2, 1))
    else
        error("Dataset 'prims' not found in the HDF5 file.")
    end
end


# --- Public Functions ---

"""
    load_data(filename::String) -> IharmData

Loads data from an iharm3d HDF5 dump file.

This function reads the primitive variables (`RHO`, `UU`, etc.) from the specified file,
converts them to `Float64`, and stores them in an `IharmData` object.

# Arguments
- `filename::String`: The path to the HDF5 dump file.

# Returns
- An `IharmData` object containing the simulation data.

# Throws
- An error if the file is not found or if the 'prims' dataset is missing.
"""
function load_data(filename::String)
    println("Loading data from '$filename' into 'iharm' module...")
    !isfile(filename) && error("File not found: $filename")

    primitives_data = Dict{String, Array{Float64, 3}}()
    h5open(filename, "r") do file
        for prim_name in VALID_PRIMS
            data_3d = _read_single_primitive(file, prim_name)
            if data_3d !== nothing
                # Convert to Float64 to ensure precision
                primitives_data[prim_name] = Float64.(data_3d)
            end
        end
    end

    if isempty(primitives_data)
        @warn "No primitives were loaded from file '$filename'."
    else
        dims = size(primitives_data[RHO])
        println("Data successfully loaded. Dimensions (N1, N2, N3): $dims")
    end
    return IharmData(primitives_data)
end

"""
    get_p(data::IharmData) -> Dict{String, Function}

Creates and returns the accessor function dictionary (`p`) from an `IharmData` object.

Each key in the returned dictionary is a primitive name (e.g., `iharm.RHO`),
and the value is a function that allows indexing and slicing of the corresponding
data.

# Arguments
- `data::IharmData`: An object containing the simulation data, usually created by `load_data`.

# Returns
- A `Dict{String, Function}` that serves as the data access interface.
"""
function get_p(data::IharmData)
    p = Dict{String, Function}()
    for prim_name in keys(data.primitives)
        # Create an anonymous function that captures the specific data array.
        # It accepts multiple indexing arguments (indices...) and passes them
        # directly to the array indexing.
        p[prim_name] = (indices...) -> data.primitives[prim_name][indices...]
    end
    println("\nAccessor dictionary 'p' created. Functions are ready to use.")
    println("Example usage for a point: p[RHO](i, j, k)")
    println("Example usage for a slice: p[RHO](:, j, :)")
    return p
end


"""
    view_slice(slice_data::AbstractArray)

Displays a 1D or 2D slice of data in a formatted way in the terminal. Useful for
quick inspection of simulation data.
"""
function view_slice(slice_data::AbstractArray)
    if ndims(slice_data) > 2
        println("The view_slice function only supports 1D or 2D arrays.")
        return
    end

    println("\n" * "="^60)
    println("Displaying slice with dimensions: ", size(slice_data))
    println("="^60)
    
    # Base.showarray is the internal function Julia uses to display arrays nicely
    Base.showarray(stdout, slice_data, false)
    println()
    println("="^60)
end

end # End of module iharm


