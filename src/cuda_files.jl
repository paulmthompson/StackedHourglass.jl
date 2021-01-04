


#adapted from https://stackoverflow.com/questions/51038294/resize-image-using-nearest-neighborhood-with-cuda

function _CUDA_resize(pIn,pOut)
    _CUDA_resize(pIn,pOut,size(pIn,1),size(pIn,2),size(pOut,1),size(pOut,2))
end

function _CUDA_resize(pIn,pOut,w_in,h_in,w_out,h_out)

    index_i = blockDim().y * (blockIdx().y-1) + threadIdx().y
    index_j = blockDim().x * (blockIdx().x-1) + threadIdx().x

    stride_i = blockDim().y * gridDim().y
    stride_j = blockDim().x * gridDim().x

    for j=index_j:stride_j:h_out
        for i=index_i:stride_i:w_out
            jIn = div(j*h_in, h_out)
            iIn = div(i*w_in, w_out)
            @inbounds pOut[i,j] = pIn[iIn,jIn]
        end
    end
    return
end

function CUDA_resize(pIn,pOut)
    @static if VERSION > v"1.5-"
        #CUDA.@sync @cuda threads=(16,16) _CUDA_resize(pIn,pOut)
        CuArrays.@sync @cuda threads=(16,16) _CUDA_resize(pIn,pOut)
    else
        CuArrays.@sync @cuda threads=(16,16) _CUDA_resize(pIn,pOut)
    end

end

function _CUDA_resize4(pIn,pOut,w_in,h_in,w_out,h_out,n)

    index_i = blockDim().y * (blockIdx().y-1) + threadIdx().y
    index_j = blockDim().x * (blockIdx().x-1) + threadIdx().x

    stride_i = blockDim().y * gridDim().y
    stride_j = blockDim().x * gridDim().x

    for k=1:n
        for j=index_j:stride_j:h_out
            for i=index_i:stride_i:w_out
                jIn = div(j*h_in, h_out)
                iIn = div(i*w_in, w_out)
                @inbounds pOut[i,j,1,k] = pIn[iIn,jIn,1,k]
            end
        end
    end
    return
end

function _CUDA_resize4(pIn,pOut)
    _CUDA_resize4(pIn,pOut,size(pIn,1),size(pIn,2),size(pOut,1),size(pOut,2),size(pIn,4))
end

function CUDA_resize4(pIn,pOut)
    @static if VERSION > v"1.5-"
        #CUDA.@sync @cuda threads=(16,16) _CUDA_resize4(pIn,pOut)
        CuArrays.@sync @cuda threads=(16,16) _CUDA_resize4(pIn,pOut)
    else
        CuArrays.@sync @cuda threads=(16,16) _CUDA_resize4(pIn,pOut)
    end
end

function _CUDA_normalize_images(pIn,meanImg,h_out,w_out,n)
    index_i = blockDim().y * (blockIdx().y-1) + threadIdx().y
    index_j = blockDim().x * (blockIdx().x-1) + threadIdx().x

    stride_i = blockDim().y * gridDim().y
    stride_j = blockDim().x * gridDim().x

    for k=1:n
        for j=index_j:stride_j:h_out
            for i=index_i:stride_i:w_out
                @inbounds pIn[i,j,1,k] = pIn[i,j,1,k] / 255
                @inbounds pIn[i,j,1,k] = pIn[i,j,1,k] - meanImg[i,j]
            end
        end
    end
    return
end

function _CUDA_normalize_images(pIn,meanImg)
    _CUDA_normalize_images(pIn,meanImg,size(pIn,1),size(pIn,2),size(pIn,4))
end

function CUDA_normalize_images(pIn,meanImg)
    @static if VERSION > v"1.5-"
        #CUDA.@sync @cuda threads=256 _CUDA_normalize_images(pIn,meanImg)
        CuArrays.@sync @cuda threads=256 _CUDA_normalize_images(pIn,meanImg)
    else
        CuArrays.@sync @cuda threads=256 _CUDA_normalize_images(pIn,meanImg)
    end
end

function CUDA_preprocess(pIn,pOut)
    n = size(pIn,4)

    w_in=size(pIn,1)
    h_in=size(pIn,2)
    w_out=size(pOut,1)
    h_out=size(pOut,2)
    @static if VERSION > v"1.5-"
        CUDA.@sync begin
            @cuda threads=(16,16) _CUDA_preprocess(pIn,pOut,w_in,h_in,w_out,h_out,n)
        end
    else
        CuArrays.@sync begin
            @cuda threads=(16,16) _CUDA_preprocess(pIn,pOut,w_in,h_in,w_out,h_out,n)
        end
    end
end

function _CUDA_preprocess(pIn,pOut,w_in,h_in,w_out,h_out,n)
    index_i = blockDim().y * (blockIdx().y-1) + threadIdx().y
    index_j = blockDim().x * (blockIdx().x-1) + threadIdx().x

    stride_i = blockDim().y * gridDim().y
    stride_j = blockDim().x * gridDim().x

    for k=1:n
        for j=index_j:stride_j:h_out
            for i=index_i:stride_i:w_out
                jIn = div(j*h_in, h_out)
                iIn = div(i*w_in, w_out)
                @inbounds pOut[j,i,1,k] = Float32(pIn[iIn,jIn,1,k])
            end
        end
    end
    return
end
