

function image_augmentation(im,ll)

    im_out = zeros(Float32,size(im,1),size(im,2),size(im,3),size(im,4)*8)
    l_out = zeros(Float32,size(ll,1),size(ll,2),size(ll,3),size(ll,4)*8)

    count=1

    for i=1:size(im,4)
        im_out[:,:,:,count] = im[:,:,:,i]
        l_out[:,:,:,count] = ll[:,:,:,i]
        count += 1
        #rotations
        for j in [pi/6, pi/2, pi, 3*pi/2, -pi/6]
            for k=1:size(im,3)
                im_out[:,:,k,count] = imrotate(im[:,:,k,i],j,Reflect())[1:size(im,1),1:size(im,2)]
            end
            for k=1:size(ll,3)
                l_out[:,:,k,count] = imrotate(ll[:,:,k,i],j,Reflect())[1:size(ll,1),1:size(ll,2)]
            end
            count += 1
        end

        #Flip x
        im_out[:,:,:,count] = reverse(im[:,:,:,i],dims=1)
        l_out[:,:,:,count] = reverse(ll[:,:,:,i],dims=1)
        count+=1

        #Flip Y
        im_out[:,:,:,count] = reverse(im[:,:,:,i],dims=2)
        l_out[:,:,:,count] = reverse(ll[:,:,:,i],dims=2)
        count+=1

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
