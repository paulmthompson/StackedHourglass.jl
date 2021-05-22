
mutable struct CUDA_Resize
    input::CuArray{Float32,4}

    xpass::CuArray{Float32,4}
    ypass::CuArray{Float32,4}
    gauss_x::CuArray{Float32,1}
    gauss_y::CuArray{Float32,1}

    resized::CuArray{Float32,4}
    output::CuArray{Float32,4}
end

function CUDA_Resize(in_w,in_h,out_w,out_h,n)
    input = convert(CuArray,zeros(Float32,in_w,in_h,1,n))

    xpass = convert(CuArray,zeros(Float32,in_w,in_h,1,n))
    ypass = convert(CuArray,zeros(Float32,in_w,in_h,1,n))
    (gauss_x,gauss_y) = get_xy_gauss_cu((in_w,in_h),(out_w,out_h))

    resized = convert(CuArray,zeros(Float32,out_w,out_h,1,n))
    output = convert(CuArray,zeros(Float32,out_w,out_h,1,n))
end

function lowpass_resize(cr::CUDA_Resize,input::AbstractArray{T,4}) where T

    cr.input[:] = convert(CuArray,input)

    CUDA_blur_x(cr.input,cr.xpass,cr.gauss_x)
    CUDA_blur_y(cr.xpass,cr.ypass,cr.gauss_y)
    CUDA_resize4(cr.ypass,cr.resized)
    CUDA_flip_xy(cr.resized,cr.output)

    nothing
end
