module StackedHourglass

#Deep Learning Libraries
using CuArrays, CuArrays.CUFFT, CUDAnative, Knet

using Images, Distributed, MAT

#exported types
export HG2

#exported methods
export subpixel, set_testing, save_hourglass, load_hourglass, features

abstract type NN end;
const HGType = Union{KnetArray{Float32,4},AutoGrad.Result{KnetArray{Float32,4}}}

include("residual.jl")
include("hourglass.jl")

include("cuda_files.jl")
include("helper.jl")
include("load.jl")
include("subpixel.jl")
include("image_preprocessing.jl")





end
