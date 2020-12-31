

mutable struct Out_Layer <: NN
    r1::Residual
    w::Param{KnetArray{Float32,4}} #weights
    b::Param{KnetArray{Float32,4}} #biases
    ms::KnetMoment #batch norm moments
    bn_p::Param{KnetArray{Float32,1}} #batch norm parameters
    training::Bool
end

function (c::Out_Layer)(x::HGType)
    r1=c.r1(x)
    out1 = conv4(c.w,r1); myfree(r1);
    out2 = out1 .+ c.b; myfree(out1);
    bn=batchnorm(out2,c.ms,c.bn_p,training=c.training); myfree(out2);
    out3 = relu.(bn);

     myfree(bn)

    out3
end

function Out_Layer(in_dim)
    r1=Residual(in_dim,in_dim)
    bn_p = Param(convert(KnetArray{Float32,1},bnparams(in_dim)))
    ms = bnmoments()
    w = Param(convert(KnetArray,xavier_normal(Float32,1,1,in_dim,in_dim)))
    b = Param(convert(KnetArray,xavier_normal(Float32,1,1,in_dim,1)))
    Out_Layer(r1,w,b,ms,bn_p,true)
end

function set_testing(f::Out_Layer,training=false)
    set_testing(f.r1,training)
    f.training=training
    nothing
end

struct FirstBlock <: NN
    c1::Conv1
    r1::Residual_skip
    p1::Pool
    r2::Residual
    r3::Residual_skip
end
FirstBlock(N)=FirstBlock(Conv1(3,64,7,2,3),Residual_skip(64,128),Pool(),Residual(128,128),Residual_skip(128,N))

function (f::FirstBlock)(x::HGType)
    c1=f.c1(x)
    r1=f.r1(c1); myfree(c1);
    p1=f.p1(r1); myfree(r1);
    r2=f.r2(p1); myfree(p1);
    r3=f.r3(r2); myfree(r2)

    r3
end

function set_testing(f::FirstBlock,training=false)
    set_testing(f.c1,training)
    set_testing(f.r1,training)
    set_testing(f.r2,training)
    set_testing(f.r3,training)
    nothing
end

struct Hourglass <: NN
    n::Int64
    up1::Residual
    pool1::Pool
    low1::Residual
    low2::Union{Hourglass,Residual} #This could be a residual or another Hourglass
    low3::Residual
    up2::Unpool
end

function Hourglass(n,f_num)
    up1 = Residual(f_num,f_num)
    pool1 = Pool()
    low1 = Residual(f_num,f_num)
    if n > 1
        low2 = Hourglass(n-1,f_num)
    else
        low2 = Residual(f_num,f_num)
    end
    low3 = Residual(f_num,f_num)
    up2 = Unpool()
    Hourglass(n,up1,pool1,low1,low2,low3,up2)
end

function (h::Hourglass)(x::HGType)

    up1 = h.up1(x)
    pool1 = h.pool1(x)
    low1 = h.low1(pool1); myfree(pool1);
    low2 = h.low2(low1); myfree(low1);
    low3 = h.low3(low2); myfree(low2);
    up2 = h.up2(low3); myfree(low3);
    out = up1 .+ up2

    myfree(up1); myfree(up2)

    out
end

function set_testing(f::Hourglass,training=false)
    set_testing(f.up1,training)
    set_testing(f.low1,training)
    set_testing(f.low2,training)
    set_testing(f.low3,training)
    nothing
end

struct HG2 <: NN
    nstack::Int64
    fb::FirstBlock
    hg::Array{Hourglass,1}
    o1::Array{Out_Layer,1}
    c1::Array{Conv0,1}
    merge_features::Array{Conv0,1}
    merge_preds::Array{Conv0,1}
end

function HG2(N,K,nstack)
    fb = FirstBlock(N)

    hg=[Hourglass(4,N) for i=1:nstack];
    o1=[Out_Layer(N) for i=1:nstack];
    c1=[Conv0(N,K,1,1,0) for i=1:nstack];
    merge_features=[Conv0(N,N,1,1,0) for i=1:(nstack-1)]
    merge_preds=[Conv0(K,N,1,1,0) for i=1:(nstack-1)]
    HG2(nstack,fb,hg,o1,c1,merge_features,merge_preds)
end

features(hg::HG2)=size(hg.merge_preds[1].w,3)

function (h::HG2)(x::HGType)
    temp=h.fb(x)

    preds=Array{HGType,1}() #Can this be typed to be the same as input?
    temps=Array{HGType,1}(undef,h.nstack) #Can this be typed to be the same as input?
    temps[1]=temp
    for i=1:h.nstack
        hg=h.hg[i](temps[i])
        features=h.o1[i](hg)
        pred=h.c1[i](features)
        push!(preds,pred)
        if i<h.nstack
            m_features = h.merge_features[i](features)
            m_preds = h.merge_preds[i](pred)
            temp1 = m_features + m_preds
            temps[i+1] = temp1 + temps[i]
            myfree(m_features); myfree(m_preds); myfree(temp1)
        end

        myfree(hg); myfree(features);
    end
    myfree(temp)
    for i=1:h.nstack
        myfree(temps[i])
    end

    preds
end

function (h::HG2)(x,y)
    preds=h(x)
    loss=0.0
    for i=1:h.nstack
        loss+=pixel_mse(y,preds[i])
    end
    loss
end
(h::HG2)(d::Knet.Data) = mean(h(x,y) for (x,y) in d)

function set_testing(f::HG2,training=false)
    set_testing(f.fb,training)
    for i=1:f.nstack
        set_testing(f.hg[i],training)
        set_testing(f.o1[i],training)
    end
    nothing
end
