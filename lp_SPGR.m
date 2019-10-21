function [Obj,it]=lp_SPGR(X, y, lss, svrg, p, lambda, dlta, eta, K, b, m, w0, idx, maxT)

% SPG with SARAH/SPIDER algorithm
% svrg: "0" for online setting; "1" for finite-sum setting
% m/M: constant large/small mini-batch size
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



for k=1:K
    
    st1 = tic;
    st2 = cputime;

    if svrg == 1 
        m = ceil(sqrt(n));        
        M = n;  
        gradG = g_gradfull(X,y,w,lss,dlta);  
        t = m;
    elseif svrg == 0 
        if b == 0 
            M = m^2;
            i = idx(count:count+M-1);
            count=count+M; 
        end
        if b > 0 
            m = ceil((b*k));
            M = m^2;
            i = idx(count:count+M-1);
            count=count+M; 
        end  
        gradG = g_gradfull(X(:,i),y(i),w,lss,dlta); 
        t = m;
    end
    
    w_old = w;
    w = w - eta*gradG;
    if p<0
        w = lp_map(w,lambda,p);
    end
    if p >= 0
        w = lp_map(w,eta*lambda,p);
    end

        
    for tau = 1:t
            i = idx(count:count+m-1);
            count=count+m;
            gradG1 = g_gradfull(X(:,i),y(i),w_old,lss,dlta);
            gradG2 = g_gradfull(X(:,i),y(i),w,lss,dlta);
            gradG = gradG2 - gradG1 + gradG;
            w_old = w;
            w = w - eta*gradG;
	    if p<0
        	w = lp_map(w,lambda,p);
    	    end
    	    if p>=0
        	w = lp_map(w,eta*lambda,p);
    	    end
    end
        
         
    T = T + M + t*m;    
    Tcpu = Tcpu + (cputime-st2);
    Time = Time + toc(st1);
    
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

it = [0,0,0; it];
Obj = [eps0; Obj];




