module StackedHourglass

#Deep Learning Libraries
using CuArrays, CuArrays.CUFFT, CUDAnative, Knet

using Images, Distributed

abstract type NN end;

include("residual.jl")
include("hourglass.jl")

include("cuda_files.jl")
include("helper.jl")
include("load.jl")
include("subpixel.jl")
include("image_preprocessing.jl")





end
