function [Obj,it]=lp_SPG(X, y, lss, p, lambda, dlta, eta, t, K, b, m, w0, idx, maxT)

% MB-SPG algorithm
% m: constant mini-batch size
% b: a constant for increasing mini-batch sizes

% p: lp regularizer (0<= p <1)
% lss = 6; % non-linear least square loss with sigmod function
% lambda: regulerizer parameter
% eta: O(1/L) step-size
% w0: initial solution
% dlta: parameter for huber loss only
% t: iteration number per stage k
% K: stage number


[d,n]=size(X);

if isempty(w0)
    w0=zeros(1,d);
end

w=w0;
if p == 0
    eps0=g_obj(X,y,lss,w0,dlta) + lambda*nnz(w0);
end
if p > 0 
    eps0=g_obj(X,y,lss,w0,dlta) + lambda*norm(w0,p);
end
if p < 0
    eps0=g_obj(X,y,lss,w0,dlta);
end




disp(sprintf('epoch=%d, obj=%.15f', 0, eps0));
Obj=[];
it=[];
T = 0;
Time = 0;
Tcpu = 0;
count = 1;


for tau=1:K*t
    
    st1 = tic;
    st2 = cputime;

    if b <= 0
        i = idx(count:count+m-1);
        count=count+m; 
    end
    if b > 0 
       m = ceil(b*tau);
       i = idx(count:count+m-1);
       count=count+m; 
    end
    
    gradG = g_gradfull(X(:,i),y(i),w,lss,dlta);
    w = w - eta*gradG;
    if p<0
	w = lp_map(w,lambda,p);
    end
    if p >= 0
        w = lp_map(w,eta*lambda,p);
    end
    Tcpu = Tcpu + (cputime-st2);
    Time = Time + toc(st1);
    T = T + m;
    
    if mod(tau, t)==0
      k = tau/t;
      it(k,1) = Tcpu; 
      it(k,2) = Time; 
      it(k,3) = T; 
      
      if p == 0
        Obj(k,1) = g_obj(X,y,lss,w,dlta) + lambda*nnz(w);
      end
      if p > 0 
    	Obj(k,1) = g_obj(X,y,lss,w,dlta) + lambda*norm(w,p);
      end
      if p < 0
        Obj(k,1) = g_obj(X,y,lss,w,dlta);
      end
      disp(sprintf('epoch=%d, obj=%.15f, T=%d, cpu=%d, time=%d', ...
            k, Obj(k,1), ceil(T),ceil(Tcpu),ceil(Time)));
      if T >= length(idx) | T > maxT
           break
      end  
    end


end

it = [0,0,0; it];
Obj = [eps0; Obj];




