
function pixel_mse(truth::Union{KnetArray{Float32,4},CuArray{Float32,4}},pred::HGType)
    loss = sum((pred .- truth).^2)
    loss / (size(pred,3) * size(pred,4))
end

function myfree(x::AutoGrad.Result)
    #myfree(x.value)
end

function myfree(x::KnetArray)
    Knet.KnetArrays.freeKnetPtr(x.ptr)
end

function myfree(x::CuArray)
end

function myfree(x::Array)
end

function gaussian_2d(x,y,x0,y0,sig_x=1,sig_y=1)
    out=[exp.(-1/2 .* ((xi .- x0).^2 ./sig_x^2 + (yi .- y0).^2 ./sig_y^2)) for xi in x, yi in y]
    out./sum(out)
end

function gaussian_1d(x,x0,sig)
    [1/(sig*sqrt(2*pi))*exp.(-1/2 .* ((xi .- x0).^2 ./sig^2)) for xi in x]
end


function calculate_sigma((in_x,in_y),(out_x,out_y))
    (0.75 * in_x/out_x , 0.75 * in_y/out_y)
end

function get_xy_gauss_cu((in_w,in_h),(out_w,out_h))

    (sig_x,sig_y) = StackedHourglass.calculate_sigma((in_w,in_h),(out_w,out_h))
    gauss_x = StackedHourglass.gaussian_1d(collect(-4:4),0,sig_x)
    gauss_x_cu = convert(CuArray,gauss_x);
    gauss_y = StackedHourglass.gaussian_1d(collect(-4:4),0,sig_y)
    gauss_y_cu = convert(CuArray,gauss_y);

    (gauss_x_cu, gauss_y_cu)
end

function create_padded_kernel(size_x,size_y,kl)
    kernel = gaussian_2d(collect(-kl:1:kl),collect(-kl:1:kl),0,0)
    kernel = kernel ./ maximum(kernel)
    kernel_pad = zeros(Float32,size_x,size_y)
    kernel_pad[(div(size_x,2)-kl):(div(size_x,2)+kl),(div(size_y,2)-kl):(div(size_y,2)+kl)] = kernel
    kernel_pad
end

#https://github.com/JuliaImages/ImageTransformations.jl/blob/master/src/resizing.jl#L239
function lowpass_filter_resize(img::AbstractArray{T,2},sz::Tuple) where T

    kern = make_gaussian_kernel(size(img),sz)
    imgr = imresize(imfilter(img, kern, NA()), sz) #Can include method here in newest ImageTransformations
end

function lowpass_filter_resize!(img::AbstractArray{T,2},output::AbstractArray{T,2},kern) where T

    imfilter!(img, img,kern, NA())
    ImageTransformations.imresize!(output, img) #Can include method here in newest ImageTransformations
    nothing
end

function make_gaussian_kernel(in_sz::Tuple,out_sz::Tuple)
    σ = map((o,n)->0.75*o/n, in_sz, out_sz)
    KernelFactors.gaussian(σ)
end

function low_pass_pyramid(im::AbstractArray{T,2},sz::Tuple) where T
    new_sz = (div(size(im,1),2),div(size(im,2),2))
    im2 = deepcopy(im)
    while ((new_sz[1] > sz[1])&&(new_sz[2] > sz[2]))
        im2=lowpass_filter_resize(im2,(new_sz))
        #im2 = im2 ./ maximum(im2)
        new_sz = (div(new_sz[1],2),div(new_sz[2],2))
    end
    im2=lowpass_filter_resize(im2,(sz))
end

function upsample_pyramid(im::AbstractArray{T,2},sz::Tuple) where T
    new_sz = (size(im,1)*2,size(im,2)*2)
    im2 = deepcopy(im)
    while ((new_sz[1] <= sz[1])&&(new_sz[2] <= sz[2]))
        im2=lowpass_filter_resize(im2,(new_sz))
        new_sz = (new_sz[1]*2,new_sz[2]*2)
    end
    im2=lowpass_filter_resize(im2,(sz))
end

function predict_single_frame(hg,img::AbstractArray{T,2},atype=KnetArray) where T

    temp_frame = convert(Array{Float32,2},img)
    temp_frame = convert(Array{Float32,2},lowpass_filter_resize(temp_frame,(256,256)))

    temp_frame = convert(atype,reshape(temp_frame,(256,256,1,1)))

    my_features = features(hg)

    set_testing(hg,false) #Turn off batch normalization for prediction
    myout=hg(temp_frame)[4]
    set_testing(hg,true) #Turn back on

    myout=convert(Array{Float32,4},myout)
end

#=
Convert Discrete points to heatmap for deep learning
=#
function make_heatmap_labels(han,real_w=640,real_h=480,label_img_size=64)

    d_points=make_discrete_all_whiskers(han)

    labels=zeros(Float32,label_img_size,label_img_size,size(d_points,1),size(han.woi,1))

    for i=1:size(labels,4)
        for j=1:size(labels,3)
            this_x = d_points[j,1,i] / real_w * label_img_size
            this_y = d_points[j,2,i] / real_h * label_img_size

            if (this_x !=0.0)&(this_y != 0.0)
                labels[:,:,j,i] = WhiskerTracking.gaussian_2d(1.0:1.0:label_img_size,1.0:1.0:label_img_size,this_y,this_x)
                labels[:,:,j,i] = labels[:,:,j,i] ./ maximum(labels[:,:,j,i])
            end
        end
    end

    labels
end

function make_heatmap_labels_keypoints(points::Array{Tuple,1},input_hw::Tuple,labels=zeros(Float32,64,64,length(points),1))

    for j=1:size(points,1)
        this_x = points[j][1] / input_hw[1] * size(labels,1)
        this_y = points[j][2] / input_hw[2] * size(labels,2)

        if (this_x !=0.0)&(this_y != 0.0)
            labels[:,:,j,1] = gaussian_2d(1.0:1.0:size(labels,1),1.0:1.0:size(labels,2),this_x,this_y)
        end
    end
end

function get_labeled_frames(han,out_hw=256,h=480,w=640,frame_rate=25)

    imgs=zeros(Float32,out_hw,out_hw,1,length(han.frame_list))

    temp=zeros(UInt8,w,h)
    for i=1:length(han.frame_list)
        frame_time = han.frame_list[i] / frame_rate
        WhiskerTracking.load_single_frame(frame_time,temp,han.wt.vid_name)
        imgs[:,:,1,i]=Images.imresize(temp',(out_hw,out_hw))
    end
    imgs
end


function batch_predict(hg::StackedHourglass.NN,input_images::CuArray{T,4},sub_input_images,
    input_f,return_ind=4,batch_size=32,batch_per_load=4) where T

    batch_predict(hg,KnetArray(input_images),sub_input_images,input_f,return_ind,batch_size,batch_per_load)
end

function batch_predict(hg::StackedHourglass.NN,input_images::KnetArray{T,4},sub_input_images::KnetArray{T,4},
        input_f::KnetArray{T,4},return_ind=4,batch_size=32,batch_per_load=4) where T

    input_hw=size(input_images,1)
    output_hw=size(input_f,1)
    my_features=size(input_f,3)

    for k=0:(batch_per_load-1)
        copyto!(sub_input_images,1,input_images,k*input_hw*input_hw*batch_size+1,input_hw*input_hw*batch_size)
        myout=hg(sub_input_images)
        copyto!(input_f,k*output_hw*output_hw*my_features*batch_size+1,myout[return_ind],1,length(myout[return_ind]))
        for kk=1:length(myout)
            Knet.KnetArrays.freeKnetPtr(myout[kk].ptr)
        end
    end

    nothing
end
