module StackedHourglass


using CUDA, Knet
const KnetMoment = Knet.Ops20.BNMoments

using Knet.CuArrays

#standard Library
using Distributed, Random

using Images, MAT, FFTW

#exported types
export HG2

#exported methods
export subpixel, set_testing, save_hourglass, load_hourglass, features

abstract type NN end;
const HGType = Union{KnetArray{Float32,4},AutoGrad.Result{KnetArray{Float32,4}},CuArray{Float32,4},AutoGrad.Result{CuArray{Float32,4}}}

const PType1 = Union{Param{KnetArray{Float32,1}},Param{CuArray{Float32,1}}}
const PType4 = Union{Param{KnetArray{Float32,4}},Param{CuArray{Float32,4}}}

include("residual.jl")
include("hourglass.jl")

include("gaussian_pyramids.jl")
include("cuda_files.jl")
include("helper.jl")
include("load.jl")
include("subpixel.jl")
include("image_preprocessing.jl")





end
