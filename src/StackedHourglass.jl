module StackedHourglass

#Deep Learning Libraries
#if VERSION > v"1.5-"
    #using CUDA, Knet
    #using CuArrays, CuArrays.CUFFT, CUDAnative, Knet
    #const KnetMoment = Knet.Ops20.BNMoments
#else
    using CuArrays, CuArrays.CUFFT, CUDAnative, Knet
    const KnetMoment = Knet.BNMoments
#end

#standard Library
using Distributed, Random

using Images, MAT, FFTW

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
