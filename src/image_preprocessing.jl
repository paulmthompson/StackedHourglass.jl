
#=
Image augmentation will apply various transformations to the training images
and accompanying labels
=#

mutable struct ImageAugmentation
    rotation::Bool
    rotation_angles::Array{Float64,1}
    flip_x::Bool
    flip_y::Bool
    set_bw::Bool
end

function ImageAugmentation()

    default_rotations = [pi/6, pi/2, pi, 3*pi/2, -pi/6]
    ImageAugmentation(true,default_rotations,true,true,true)
end

function rotation_augmentation(im,ll,im_out,l_out,ind,aug::ImageAugmentation,count)
    for j in aug.rotation_angles
        for k=1:size(im,3)
            im_out[:,:,k,count] = imrotate(im[:,:,k,ind],j,ImageTransformations.Flat())[1:size(im,1),1:size(im,2)]
        end
        for k=1:size(ll,3)
            l_out[:,:,k,count] = imrotate(ll[:,:,k,ind],j,0)[1:size(ll,1),1:size(ll,2)]
        end
        count += 1
    end
    count
end

function flip_x_augmentation(im,ll,im_out,l_out,ind,aug::ImageAugmentation,count)
    im_out[:,:,:,count] = reverse(im[:,:,:,ind],dims=1)
    l_out[:,:,:,count] = reverse(ll[:,:,:,ind],dims=1)
    count+=1
end

function flip_y_augmentation(im,ll,im_out,l_out,ind,aug::ImageAugmentation,count)
    im_out[:,:,:,count] = reverse(im[:,:,:,ind],dims=2)
    l_out[:,:,:,count] = reverse(ll[:,:,:,ind],dims=2)
    count+=1
end

function random_block(xdim,ydim,max_length)

    x1 = rand(1:div(xdim,2))
    x2 = rand(x1:(x1+max_length))
    y1 = rand(1:div(ydim,2))
    y2 = rand(y1:(y1+max_length))

    (x1,x2,y1,y2)
end

function flip_bw_augmentation(im,ll,im_out,l_out,ind,aug::ImageAugmentation,count)

    #Random Black
    (x1,x2,y1,y2) = random_block(size(im,1),size(im,2),5)
    im_out[:,:,:,count] = im[:,:,:,ind]
    l_out[:,:,:,count] = ll[:,:,:,ind]
    im_out[x1:x2,y1:y2,:,count] .= 0.0
    count+=1

    #Random White
    (x1,x2,y1,y2) = random_block(size(im,1),size(im,2),5)
    im_out[:,:,:,count] = im[:,:,:,ind]
    l_out[:,:,:,count] = ll[:,:,:,ind]
    im_out[x1:x2,y1:y2,:,count] .= 1.0
    count+=1
end

function image_augmentation(im,ll,aug::ImageAugmentation)

    rot_size = 0
    flip_x_size = 0
    flip_y_size = 0
    set_bw_size = 0
    if aug.rotation
        rot_size += length(aug.rotation_angles)
    end
    if aug.flip_x
        flip_x_size = 1
    end
    if aug.flip_y
        flip_y_size = 1
    end
    if aug.set_bw
        set_bw_size = 2
    end
    array_size = rot_size + flip_x_size + flip_y_size + set_bw_size + 1

    im_out = zeros(Float32,size(im,1),size(im,2),size(im,3),size(im,4) * array_size)
    l_out = zeros(Float32,size(ll,1),size(ll,2),size(ll,3),size(ll,4) * array_size)

    count=1

    for i=1:size(im,4)
        im_out[:,:,:,count] = im[:,:,:,i]
        l_out[:,:,:,count] = ll[:,:,:,i]
        count += 1
        #rotations
        if aug.rotation
            count = rotation_augmentation(im,ll,im_out,l_out,i,aug,count)
        end

        #Flip x
        if aug.flip_x
            count = flip_x_augmentation(im,ll,im_out,l_out,i,aug,count)
        end

        #Flip Y
        if aug.flip_y
            count = flip_y_augmentation(im,ll,im_out,l_out,i,aug,count)
        end

        if aug.set_bw
            count = flip_bw_augmentation(im,ll,im_out,l_out,i,aug,count)
        end

        # 0.75 Scale (zoom in)

        #1.25 Scale (Reflect around places where image is empty)
    end

    #Shuffle Positions
    myinds=Random.shuffle(collect(1:size(im_out,4)));
    im_out=im_out[:,:,:,myinds]
    l_out=l_out[:,:,:,myinds]

    (im_out,l_out)
end

function normalize_images(ii)

    mean_img = mean(ii,dims=4)[:,:,:,1]
    std_img = std(ii,dims=4)[:,:,:,1]
    std_img[std_img .== 0.0] .= 1

    ii = (ii .- mean_img) ./ std_img

    min_ref = minimum(ii)
    ii = ii .- min_ref

    max_ref = maximum(ii)
    ii = ii ./ max_ref

    (mean_img,std_img,min_ref,max_ref)
end

function normalize_new_images(ii::KnetArray,mean_img::Array,std_img,min_ref,max_ref)
    normalize_new_images(ii,convert(KnetArray,mean_img),convert(KnetArray,std_img),min_ref,max_ref)
end

function normalize_new_images(ii,mean_img)
    ii = ii ./ 255
    ii = ii .- mean_img
end
