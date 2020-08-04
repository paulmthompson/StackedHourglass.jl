module StackedHourglass

#Deep Learning Libraries
using CuArrays, CuArrays.CUFFT, CUDAnative, Knet

include("residual")
include("hourglass.jl")

include("cuda_files.jl")
include("helper.jl")
include("load.jl")
include("subpixel")





end
