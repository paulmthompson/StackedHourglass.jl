

mutable struct Conv1 <: NN
    w::PType4 #weights
    b::PType4 #biases
    ms::KnetMoment #batch norm moments
    bn_p::PType1 #batch norm parameters
    stride::Int64
    padding::Int64
    training::Bool
end

function (c::Conv1)(x::HGType)
    bn = batchnorm(x,c.ms,c.bn_p,training=c.training)
    relu_out = relu.(bn)
    output1 = conv4(c.w,relu_out,stride=c.stride,padding=c.padding)
    output2 = output1 .+ c.b

    myfree(bn); myfree(relu_out); myfree(output1)

    output2
end

function Conv1(in_dim::Int,out_dim::Int,k_s,stride,padding,atype=KnetArray)

    bn_p = bnparams(Float32,in_dim)
    bn_p = Param(convert(atype{Float32,1},bn_p))
    ms = bnmoments()

    w = xavier_normal(Float32,k_s,k_s,in_dim,out_dim)
    b = xavier_normal(Float32,1,1,out_dim,1)
    w = Param(convert(atype,w))
    b = Param(convert(atype,b))
    Conv1(w,b,ms,bn_p,stride,padding,true)
end

function set_testing(f::Conv1,training=false)
    f.training=training
    nothing
end

mutable struct Conv0 <: NN
    w::PType4 #weights
    b::PType4 #biases
    stride::Int64
    padding::Int64
end

function (c::Conv0)(x::HGType)
    out1=conv4(c.w,x,stride=c.stride,padding=c.padding)
    out2 = out1 .+ c.b

    myfree(out1)

    out2
end

function Conv0(in_dim::Int,out_dim::Int,k_s,stride,padding,atype=KnetArray)
    w = xavier_normal(Float32,k_s,k_s,in_dim,out_dim)
    b = xavier_normal(Float32,1,1,out_dim,1)
    w = Param(convert(atype,w))
    b = Param(convert(atype,b))
    Conv0(w,b,stride,padding)
end

struct Residual <: NN
    c1::Conv1
    c2::Conv1
    c3::Conv1
end

function (r::Residual)(x::HGType)
    c1=r.c1(x)
    c2=r.c2(c1)
    output = r.c3(c2) .+ x

    myfree(c1); myfree(c2)

    output
end

function Residual(in_dim::Int,out_dim::Int,atype=KnetArray)
    c1=Conv1(in_dim,div(out_dim,2),1,1,0,atype)
    c2=Conv1(div(out_dim,2),div(out_dim,2),3,1,1,atype)
    c3=Conv1(div(out_dim,2),out_dim,1,1,0,atype)
    Residual(c1,c2,c3)
end

function set_testing(f::Residual,training=false)
    set_testing(f.c1,training)
    set_testing(f.c2,training)
    set_testing(f.c3,training)
    nothing
end

mutable struct Residual_skip <: NN
    w::PType4
    b::PType4
    c1::Conv1
    c2::Conv1
    c3::Conv1
end

function (r::Residual_skip)(x::HGType)
    residual1 = conv4(r.w,x,stride=1)
    residual2 = residual1 .+ r.b
    c1=r.c1(x)
    c2=r.c2(c1)
    output = r.c3(c2) .+ residual2

    myfree(c1); myfree(c2); myfree(residual1); myfree(residual2);

    output
end

function Residual_skip(in_dim::Int,out_dim::Int,atype=KnetArray)
    c1=Conv1(in_dim,div(out_dim,2),1,1,0,atype)
    c2=Conv1(div(out_dim,2),div(out_dim,2),3,1,1,atype)
    c3=Conv1(div(out_dim,2),out_dim,1,1,0,atype)

    w = xavier_normal(Float32,1,1,in_dim,out_dim)
    b = xavier_normal(Float32,1,1,out_dim,1)
    w=Param(convert(atype,w))
    b=Param(convert(atype,b))
    Residual_skip(w,b,c1,c2,c3)
end

function set_testing(f::Residual_skip,training=false)
    set_testing(f.c1,training)
    set_testing(f.c2,training)
    set_testing(f.c3,training)
    nothing
end

struct Pool
end
(p::Pool)(x::HGType) = pool(x)

struct Unpool
end
(u::Unpool)(x::HGType) = unpool(x)
