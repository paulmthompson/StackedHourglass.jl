
function pixel_mse(truth::KnetArray{Float32,4},pred::HGType)
    loss = sum((pred .- truth).^2)
    loss / (size(pred,3) * size(pred,4))
end

function myfree(x::AutoGrad.Result)
    #myfree(x.value)
end

function myfree(x::KnetArray)
    Knet.freeKnetPtr(x.ptr)
end

gaussian_2d(x,y,x0,y0)=[1/sqrt.(2 .* pi) .* exp.(-1 .* ((xi .- x0).^2 + (yi .- y0).^2)) for xi in x, yi in y]

function create_padded_kernel(size_x,size_y,kl)
    kernel = gaussian_2d(collect(-kl:1:kl),collect(-kl:1:kl),0,0)
    kernel = kernel ./ maximum(kernel)
    kernel_pad = zeros(Float32,size_x,size_y)
    kernel_pad[(div(size_x,2)-kl):(div(size_x,2)+kl),(div(size_y,2)-kl):(div(size_y,2)+kl)] = kernel
    kernel_pad
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
