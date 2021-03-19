


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

    c = blockIdx().y
    b = blockIdx().x
    a = threadIdx().x

    if ((a <= h_out)&&(b <= w_out))&&(c <=n)
        jIn = div(b*h_in, h_out)
        iIn = div(a*w_in, w_out)
        @inbounds pOut[a,b,1,c] = pIn[iIn,jIn,1,c]
    end

    return
end

function _CUDA_resize4(pIn,pOut)
    _CUDA_resize4(pIn,pOut,size(pIn,1),size(pIn,2),size(pOut,1),size(pOut,2),size(pIn,4))
end

function CUDA_resize4(pIn,pOut)
    numblocks_x = size(pOut,2)
    numblocks_y = size(pOut,4)

    CuArrays.@sync @cuda threads=256 blocks=(numblocks_x,numblocks_y) _CUDA_resize4(pIn,pOut)
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
        #CUDA.@sync @cuda threads=(16,16) _CUDA_preprocess(pIn,pOut,w_in,h_in,w_out,h_out,n)
        CuArrays.@sync @cuda threads=(16,16) _CUDA_preprocess(pIn,pOut,w_in,h_in,w_out,h_out,n)
    else
        CuArrays.@sync @cuda threads=(16,16) _CUDA_preprocess(pIn,pOut,w_in,h_in,w_out,h_out,n)
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


function _CUDA_blur_x(pIn,pOut,w_in,h_in,gauss,n)

    c = blockIdx().z
    b = blockIdx().y
    a = threadIdx().x

    while (a<w_in)
        if ((a <= w_in)&&(b <= h_in))&&(c <=n)
            temp = 0.0
            for yy = 1:9

                new_y = yy-5 + a
                if new_y<1
                    new_y=abs(new_y)+1
                end
                if new_y>w_in
                    new_y = w_in - (new_y - w_in)
                end

                @inbounds temp = temp + Float32(pIn[new_y,b,1,c]) * gauss[yy]
            end
            @inbounds pOut[a,b,1,c] = temp
        end
        a += blockDim().x*blockIdx().x
    end
    return nothing
end

function _CUDA_blur_x(pIn,pOut,gauss)
    _CUDA_blur_x(pIn,pOut,size(pIn,1),size(pIn,2),gauss,size(pIn,4))
end

function CUDA_blur_x(pIn,pOut,gauss)
    numblocks_x = ceil(Int,size(pOut,1)/ 256)
    numblocks_y = size(pOut,2)
    numblocks_z = size(pOut,4)
    CuArrays.@sync @cuda threads=256 blocks=(numblocks_x,numblocks_y,numblocks_z) _CUDA_blur_x(pIn,pOut,gauss)
end

function _CUDA_blur_y(pIn,pOut,gauss)
    _CUDA_blur_y(pIn,pOut,size(pIn,1),size(pIn,2),gauss,size(pIn,4))
end

function CUDA_blur_y(pIn,pOut,gauss)
    numblocks_x = ceil(Int,size(pOut,2)/ 256)
    numblocks_y = size(pOut,1)
    numblocks_z = size(pOut,4)
    CuArrays.@sync @cuda threads=256 blocks=(numblocks_x,numblocks_y,numblocks_z) _CUDA_blur_x(pIn,pOut,gauss)
end

function _CUDA_blur_y(pIn,pOut,w_in,h_in,gauss,n)

    c = blockIdx().z
    b = blockIdx().y
    a = threadIdx().x

    while (a<w_in)
        if ((a <= h_in)&&(b <= w_in))&&(c <=n)
            temp = 0.0
            for yy = 1:9

                new_y = yy-5 + a
                if new_y<1
                    new_y=abs(new_y)+1
                end
                if new_y>h_in
                    new_y = h_in - (new_y - h_in)
                end

                @inbounds temp = temp + Float32(pIn[b,new_y,1,c]) * gauss[yy]
            end
            @inbounds pOut[a,b,1,c] = temp
        end
        a += blockDim().x*blockIdx().x
    end
    return nothing
end
