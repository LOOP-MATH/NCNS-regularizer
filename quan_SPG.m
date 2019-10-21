function [Obj,it]=quan_SPG(X, y, Xt, yt, yorig, lss, p, dlta, eta, t, K, b, m, w0, idx, maxT,decreasing)

% MB-SPG algorithm for quantization
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
[dt,nt]=size(Xt);


w=w0;
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
    
    gradG = g_gradfull(X(:,i),y(:,i),w,lss,dlta);
    w = w - eta*gradG;
    wq = KeepDigits(w,p);
    w = (w + eta.*wq)./(1+eta); 
    Tcpu = Tcpu + (cputime-st2);
    Time = Time + toc(st1);
    T = T + m;

    if mod(tau,decreasing) == 0
         eta = eta/2;
    end
        
    if mod(tau, t)==0

      k = tau/t;
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
           
      disp(sprintf('epoch=%d, accu1=%.20f, accu2=%.20f, TRaccu1=%.20f, TRaccu2=%.20f, T=%d, cpu=%d, time=%d', ...
            k, Obj(k,1), Obj(k,2), Obj(k,3), Obj(k,4), ceil(T),ceil(Tcpu),ceil(Time)));
        
      if T >= length(idx) | T > maxT
        break
      end  
      
    end


end




