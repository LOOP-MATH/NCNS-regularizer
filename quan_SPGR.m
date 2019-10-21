function [Obj,it]=quan_SPGR(X, y, Xt, yt, yorig, lss, svrg, p, dlta, eta, K, b, m, w0, idx, maxT, decreasing)

% SPG with SARAH/SPIDER algorithm for quantization
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
[dt,nt]=size(Xt);

w=w0;
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
        gradG = g_gradfull(X(:,i),y(:,i),w,lss,dlta); 
        t = m;
    end
    
    w_old = w;
    w = w - eta*gradG;
    wq = KeepDigits(w,p);
    w = (w + eta.*wq)./(1+eta);
        
    for tau = 1:t
            i = idx(count:count+m-1);
            count=count+m;
            gradG1 = g_gradfull(X(:,i),y(:,i),w_old,lss,dlta);
            gradG2 = g_gradfull(X(:,i),y(:,i),w,lss,dlta);
            gradG = gradG2 - gradG1 + gradG;
            w_old = w;
            w = w - eta*gradG;
            wq = KeepDigits(w,p);
            w = (w + eta.*wq)./(1+eta);
            
            if mod(tau,decreasing) == 0
                eta = eta/2;
            end
            

    end
        
    T = T + M + t*m;    
    Tcpu = Tcpu + (cputime-st2);
    Time = Time + toc(st1);
    
    it(k,1) = Tcpu; 
    it(k,2) = Time; 
    it(k,3) = T; 

      yhat = wq*Xt;
      [temp,pred] = max(yhat,[],1); 
      pred = pred-1;       
      Obj(k,1) = sum(yt == pred)/nt;
      
      yhat = w*Xt;     
      [temp,pred] = max(yhat,[],1); 
      pred = pred-1;    
      Obj(k,2) = sum(yt == pred)/nt;
      
      yhat = wq*X;
      [temp,pred] = max(yhat,[],1); 
      pred = pred-1;       
      Obj(k,3) = sum(yorig == pred)/n;
      
      yhat = w*X;     
      [temp,pred] = max(yhat,[],1); 
      pred = pred-1;    
      Obj(k,4) = sum(yorig == pred)/n;
     

    disp(sprintf('epoch=%d, obj=%.20f, obj=%.20f, TRaccu1=%.20f, TRaccu2=%.20f, T=%d, cpu=%d, time=%d', ...
            k, Obj(k,1),Obj(k,2), Obj(k,3), Obj(k,4), ceil(T),ceil(Tcpu),ceil(Time)));   
        
    if T >= length(idx) | T > maxT
        break
    end  
                
end





