#using PyPlot
using Debugger
using Plots
using LinearAlgebra
# input data 
ENV["GKSwstype"] = "100" #https://discourse.julialang.org/t/plots-gr-png-results-in-too-many-open-files/48998/10
Plots.gr()


# define network size 
d = 1; # dim inputs 
m = 20; # nr hidden units  
n = 30;  # nr of input

mutable struct Model
    d::Integer  # input dimension
    m::Integer # hidden dimension
    A::Array{Float64, 2}
    b::Array{Float64, 2}
    w::Array{Float64, 2}
    σ::Function
    dσ::Function
end

function Model(d::Integer, m::Integer)
    w = .1*randn(1,m) ./ sqrt(m);
    A = .1*randn(m,d);
    b = .1*randn(m,1);
    σ = x -> max(x, 0)
    dσ = x -> (x>=0)
    Model(d, m, A, b, w, σ, dσ)
end

model = Model(d, m)

#function Base.copy(m::Model)
#    Model(m.d, m.m, m.A, m.b, m.w)
#end



x = sort(1.5*randn(d,n), dims=2); 
y = sin.(x); 
# step size 
step=.01;

# initialize net 1 to be trained with GD  
# initialize net 2 to be trained with GN 



model2 = deepcopy(model)

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




function gradient_descent!(model::Model, x, y, step::Float64=0.01, T::Int=10000; mode="normal")

    A, b, w = model.A, model.b, model.w

    lt = @layout [a b; c d]
    p = plot(ylim=(-1.25, 1.25), layout=lt)
    #p4 = scatter(color=:red, yscale=:log10, grid=true, fillto=1e-2)
    p4 = scatter()
    #T =1000;
    anim = Animation("images/$mode", String[])

    for t = 1:T
        # GRADIENT DESCENT 
        U = A*x .+ b
        V = model.σ.(U)
        # of size m x n
        r = w*V - y;  # square l2 loss
        # of size 1xn
        # Jacobian, of size d x m(d+2) 
        D = w' .* (model.dσ.(A*x .+ b))  # derivative, size m x n
        J = [V' mkron(D', x') D']   # apply kron on each line i.e. sample
        #for i=1:length(x)
        #    Jbis[i,:] = [max.(A*x[i]+b,0)', kron(w.*(A*x[i]+b.>=0)',x[i]), w.*(A*x[i]+b.>=0)']; 
        #end
        #println("norm diff:", norm(Jbis-J))
        # gradient direction 
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
            #subplot(2,2,1)
            p1 = plot(x',y', color=:blue, linestyle=:dashdot, markerstyle=:dot)
            f = w*max.(A*x .+ b,0);
            p3 = plot(x',f', color=:red, linestyle=:dashdot, markerstyle=:cross)
            #
            xx = range(-maximum(abs.(x)),maximum(abs.(x)),length=200) |> collect  # collect 
            f = w*max.(A*xx' .+ b,0);
            p2 = plot(xx,f', color=:red, linestyle=:dash)
            err =r*r'; 
            #grid(true)
            #show()
            #figure()
            #subplot(2,2,3)
            scatter!(p4, [t],err, yscale=:log10, grid=true, color=:red)        #grid(true) 
            #set(gca, "YScale", "log")
            #
            #drawnow
            #show()
            #sleep(0.001)
            p = plot(p1, p2, p3, p4)
            plot!(p, title = mode == "normal" ? "Gradient descent" : "Gauss Newton" * ", loss $err")
            frame(anim)
        end 
    end
end



function gauss_newton!(model::Model, x, y, step::Float64=0.01, T::Int=10000)

    A, b, w = model.A, model.b, model.w

    for t=1:T
    # GAUSS NEWTON 
        U = A*x .+ b2
        V2 = max.(U, 0)
        # of size m x n
        r2 = w2*V2 - y;  # square l2 loss
        # of size 1xn
        # Jacobian, of size d x m(d+2) 
        D2 = w2' .* ((A*x .+ b2) .>= 0)  # derivative, size m x n
        J2 = [V2' mkron(D2', x') D2']   # apply kron on each line i.e. sample
        # Jacobian 
        # Gauss-Newton direction 
        g2 = r*J2*inv(J2'*J2 + .1*I); 
        # update parameters 
        w2 = w2 .- step.*g2[1:m];
        A = A .- step.*reshape(g2[m+1:m+m*d],m,d);  
        b2 = b2 .- step.*g2[m+m*d+1:end]';  
        # plot 

        if mod(t,min(50,floor(T/100)))==0

            subplot(2,2,2)
            plot!(p1, x,y, color=:blue, markerstyle=:star)
            #hold 
            f = w*max.(A*x .+ b2,0);
            plot!(p3, x,f, color=:red, markerstyle=:star)
            f = w*max.(A*xx .+ b2,0);
            plot!(p2, xx,f,color=:red, linstyle=:dash)
            #ylim([-1.25,1.25])
            #figure()
            err2 =r2*r2';
            #title("Gauss-Newton, loss:$err2")
            #grid(true)

            #subplot(2,2,4)
            scatter!(p4, t,err2, color=:red)
            #hold(true)
            p = plot(p1, p2, p3, p4)
            plot!(p, title = "Gradient descent,loss $err")
        end 
    end 
end



    #function f = myfun(x,w,A,b)
    #w*max(A*x + b,0); 
    #end 

    #function J = jacob(x,w,A,b)
    #for i=1:numel(x)
    #    J(i,:) = [max(A*x + b,0), kron(w.*(A*x+b>=0),x(i)), w.*(A*x+b>=0)]; 
    #end
    #end
