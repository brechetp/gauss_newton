#using PyPlot
using Debugger
using Plots
using LinearAlgebra
using ProgressMeter
# input data 
ENV["GKSwstype"] = "100" #https://discourse.julialang.org/t/plots-gr-png-results-in-too-many-open-files/48998/10
Plots.gr()


# define network size 
d = 1; # dim inputs 
m = 20; # nr hidden units  
n = 30;  # nr of input

abstract type Model end
abstract type NeuralNetwork <: Model end

mutable struct ShallowNetwork <: NeuralNetwork
    # output dimension : 1
    d::Integer  # input dimension
    p::Integer # output dimension
    m::Integer # hidden dimension
    A::Array{Float64, 2}  # of size mxd
    b::Array{Float64, 2}  # of size mx1
    w::Array{Float64, 2}  # of size pxm
    sigma::Function
    dsigma::Function
end

function ShallowReLU(d::Integer, m::Integer)
    p = 1
    w = .1*randn(p,m) ./ sqrt(m);
    A = .1*randn(m,d);
    b = .1*randn(m,1);
    sigma = x -> max(x, 0)
    dsigma = x -> (x>=0)
    ShallowNetwork(d, p, m, A, b, w, sigma, dsigma)
end

model = ShallowReLU(d, m)
model2 = deepcopy(model)

#function Base.copy(m::Model)
#    Model(m.d, m.m, m.A, m.b, m.w)
#end



x = sort(1.5*randn(d,n), dims=2); 
y = sin.(x); 
# step size 
step=.001;

# initialize net 1 to be trained with GD  
# initialize net 2 to be trained with GN 




"""
   mkron(A, B)

Compute the vectorized kronecker product between A and B
Dimension that is equal is interpreted as the one on which to vectorize
"""
function mkron(A, B)
    m, n = size(A)
    p, q = size(B)
    if m == p  # prefer to iterate over rows
        return mkron(A', B')'
    end
    @assert(n==q)
    K = zeros(m*p, n)
    for j=1:n
        K[:, j] = kron(A[:, j], B[:, j])
    end
    return K
end




function gradient_descent!(model::ShallowNetwork, x, y; step::Float64=0.01, T::Int=10000, mode="normal")

    A, b, w = model.A, model.b, model.w
    ret =[] 
    #ret = Array{Any, 1}  # the results 

    @showprogress 1 "Gradient descent" for t = 1:T
        # GRADIENT DESCENT 
        U = A*x .+ b
        V = model.sigma.(U)
        # of size m x n
        r = w*V - y;  # square l2 loss
        # of size 1xn
        # Jacobian, of size d x m(d+2) 
        D = w' .* (model.dsigma.(A*x .+ b))  # derivative, size m x n
        J = [V' mkron(D', x') D']   # apply kron on each line i.e. sample
        g = r*J
        if mode=="gauss_newton"
            g*=inv(J'*J + .1*I)
        end
        # of size 1 x m(d+2)
        # update parameters 
        w = w - step.*g[1:m]';
        A = A .- step.*reshape(g[ m+1:m+m*d ],m,d);  
        b = b - step.*g[m+m*d+1:end];  
        # plot 
        if mod(t,min(50,floor(T/100)))==0
            f = w*model.sigma.(A*x .+ b)
            xx = range(-maximum(abs.(x)),maximum(abs.(x)),length=200) |> collect  # collect 
            ff = w*model.sigma.(A*xx' .+ b)
            push!(ret,(t=t, xx=xx, f=f', ff=ff', r=r))
        end
    end
    return ret
end



function plot_result(rets, x, y)
    p4 = scatter()
    @showprogress for ret in rets
        #subplot(2,2,1)
        p1 = plot(x',y', title="ground truth", color=:blue, linestyle=:dashdot, markershape=:star)
        p3 = plot(x',ret.f, title="prediction", color=:red, linestyle=:dashdot, markershape=:cross)
        p2 = plot(ret.xx, ret.ff, title="test data", color=:red, linestyle=:dash)
        err =ret.r*ret.r'; 
        #grid(true)
        #show()
        #figure()
        #subplot(2,2,3)
        scatter!(p4,[ret.t],err, title="Error", yscale=:log10, grid=true, color=:red)        #grid(true) 
        #set(gca, "YScale", "log")
        #
        p = plot(p1, p2, p3, p4)
        Plots.savefig("images/ret$(ret.t).pdf")
        closeall()
    end 
end
