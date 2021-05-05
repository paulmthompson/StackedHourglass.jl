
mutable struct Gaussian_Pyramid
    dims::Array{Tuple{Int64,Int64},1}
    imgs::Array{Array{Float64,2},1}
    kerns::Array{Tuple{KernelFactors.ReshapedOneD,KernelFactors.ReshapedOneD},1}
end

function Gaussian_Pyramid(in_sz::Tuple)
    Gaussian_Pyramid([in_sz],[zeros(Float64,in_sz)],
        Array{Tuple{KernelFactors.ReshapedOneD,KernelFactors.ReshapedOneD},1}())
end

function Gaussian_Pyramid(in_sz::Tuple,out_sz::Tuple)
    mygauss=Gaussian_Pyramid(in_sz)

    if (in_sz[1] * in_sz[2]) > (out_sz[1] * out_sz[2])

        println("Downsampling pyramid")

        new_sz = (div(in_sz[1],2),div(in_sz[2],2))
        #push!(mygauss.dims,new_sz)
        #push!(mygauss.imgs,zeros(Float64,new_sz))
        #push!(mygauss.kerns,make_gaussian_kernel(in_sz,new_sz))
        old_sz = in_sz

        while ((new_sz[1] > out_sz[1])&&(new_sz[2] > out_sz[2]))

            push!(mygauss.dims,new_sz)
            push!(mygauss.imgs,zeros(Float64,new_sz))
            push!(mygauss.kerns,make_gaussian_kernel(old_sz,new_sz))

            old_sz = new_sz
            new_sz = (div(new_sz[1],2),div(new_sz[2],2))

        end
        if mygauss.dims[end] != out_sz
            push!(mygauss.dims,out_sz)
            push!(mygauss.imgs,zeros(Float64,out_sz))
            push!(mygauss.kerns,make_gaussian_kernel(new_sz,out_sz))
        end

    else
        println("Upsampling pyramid")

        new_sz = (in_sz[1]*2,in_sz[2]*2)
        old_sz = in_sz

        #push!(mygauss.dims,new_sz)
        #push!(mygauss.imgs,zeros(Float64,new_sz))
        #push!(mygauss.kerns,make_gaussian_kernel(in_sz,new_sz))

        while ((new_sz[1] <= out_sz[1]) && (new_sz[2] <= out_sz[2]))

            push!(mygauss.dims,new_sz)
            push!(mygauss.imgs,zeros(Float64,new_sz))
            push!(mygauss.kerns,make_gaussian_kernel(old_sz,new_sz))
            old_sz = new_sz
            new_sz = (new_sz[1]*2,new_sz[2]*2)
        end
        if mygauss.dims[end] != out_sz
            push!(mygauss.dims,out_sz)
            push!(mygauss.imgs,zeros(Float64,out_sz))
            push!(mygauss.kerns,make_gaussian_kernel(new_sz,out_sz))
        end
    end
    mygauss
end

function lowpass_filter_resize(gauss::Gaussian_Pyramid)

    for k=1:length(gauss.kerns)
       lowpass_filter_resize!(gauss.imgs[k],gauss.imgs[k+1],gauss.kerns[k])
    end
end

function lowpass_filter_resize(gauss::Gaussian_Pyramid,input::AbstractArray{Float64,2},output::AbstractArray{Float64,2})

   gauss.imgs[1][:] = input
    lowpass_filter_resize(gauss)
    output[:] = gauss.imgs[end]

    nothing
end
