clf 
clear all
% input data 
x = sort(1.5*randn(1,30)); 
y = sin(x); 
% define network size 
n = 1; % dim inputs 
m = 20; % nr hidden units  
% step size 
step=.01;

% initialize net 1 to be trained with GD  
w = .1*randn(1,m)/sqrt(m);
A = .1*randn(m,n);
b = .1*randn(m,1);
% initialize net 2 to be trained with GN 
w2 = w; 
A2 = A; 
b2 = b; 

T =1000;
for t=1:T
% GRADIENT DESCENT 
r = w*max(A*x + b,0) -y; 
% Jacobian 
for i=1:numel(x)
    J(i,:) = [max(A*x(i)+b,0)', kron(w.*(A*x(i)+b>=0)',x(i)), w.*(A*x(i)+b>=0)']; 
end
% gradient direction 
g = r*J; 
% update parameters 
w = w - step*g(1:m);
A = A - step*reshape(g(m+1:m+m*n),m,n);  
b = b - step*g(m+m*n+1:end)';  
% plot 
if mod(t,min(50,floor(T/100)))==0
subplot(2,2,1)
plot(x,y,'*b')
hold on 
f = w*max(A*x + b,0);
plot(x,f,'*r')
%
xx = linspace(-max(abs(x)),max(abs(x)),200); 
f = w*max(A*xx + b,0);
plot(xx,f,'-r')
ylim([-1.25,1.25])
err =r*r'; 
title(strcat('Gradient descent,{ }loss:{ }',num2str(err)))
grid on
hold off
subplot(2,2,3)
scatter(t,err,'r')
grid on 
set(gca, 'YScale', 'log')
hold on 

drawnow
end 

% GAUSS NEWTON 
r2 = w2*max(A2*x + b2,0) -y; 
% Jacobian 
for i=1:numel(x)
    J2(i,:) = [max(A2*x(i)+b2,0)', kron(w2.*(A2*x(i)+b2>=0)',x(i)), w2.*(A2*x(i)+b2>=0)']; 
end
% Gauss-Newton direction 
g2 = r2*J2*inv(J2'*J2 + .1*eye(size(J2,2))); 
% update parameters 
w2 = w2 - step*g2(1:m);
A2 = A2 - step*reshape(g2(m+1:m+m*n),m,n);  
b2 = b2 - step*g2(m+m*n+1:end)';  
% plot 

if mod(t,min(50,floor(T/100)))==0

subplot(2,2,2)
plot(x,y,'*b')
hold on 
f2 = w2*max(A2*x + b2,0);
plot(x,f2,'*r')
xx = linspace(-max(abs(x)),max(abs(x)),200); 
f2 = w2*max(A2*xx + b2,0);
plot(xx,f2,'-r')
ylim([-1.25,1.25])
hold off
err2 =r2*r2';
title(strcat('Gauss-Newton,{ }loss:{ }',num2str(err2)))
grid on

subplot(2,2,4)
scatter(t,err2,'r')
hold on
grid on 
set(gca, 'YScale', 'log')
drawnow
end 


end 



%function f = myfun(x,w,A,b)
%w*max(A*x + b,0); 
%end 

%function J = jacob(x,w,A,b)
%for i=1:numel(x)
%    J(i,:) = [max(A*x + b,0), kron(w.*(A*x+b>=0),x(i)), w.*(A*x+b>=0)]; 
%end
%end

