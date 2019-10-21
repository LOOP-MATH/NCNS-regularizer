function [g,ind] = g_grad(X,y,w,lss,dlta)
% d: dimension of feature
% X: d-by-1
% y: 1-by-1
% w: 1-by-d

% lss: loss function
  % 1: 'hinge'
  % 2: 'logistic' 
  % 3: 'least square'
  % 4: 'huber'
  % 5: 'squared hinge'


ind = find(X);
g=0;

if lss == 1 % 'hinge'
   x = X(ind);
   temp = y*x;
   pred = w(ind)*temp;
   if pred < 1
     g = -temp';
   end
end

if lss == 2 % 'logistic' 
   x = X(ind);
   temp = -y*x;
   pred = w(ind)*temp;
   if pred > 37
      g = temp';
   elseif pred > -746
      Exp = exp(pred);
      g = (Exp/(1+Exp))*temp';
   end
end

if lss == 3 % 'least'
   x = X(ind);
   pred = w(ind)*x - y;
   g = pred*x';
end

if lss == 4  % 'huber'
   x = X(ind);
   pred = w(ind)*x - y;
   if abs(pred) <= dlta
      g = pred*x'; 
   else
      if pred > 0
        g = dlta*x';
      elseif pred < 0
        g = -dlta*x';
      end
   end
end

if lss == 5 % 'squared hinge'
   x = X(ind);
   temp = y*x;
   pred = w(ind)*temp;
   if pred < 1
     g =2*(pred-1)*temp';
   end
end


if lss == 6 % non-linear least square loss with sigmod function
   x = X(ind);
   pred = 1/(1+exp(-w(ind)*x));
   g =  (2*(pred-y)*pred*(1-pred)).*x';   
end

 if lss == 7 % truncated least square  
             % dlta: tuncation parameter \alpha
    x = X(ind);
    pred = w(ind)*x - y;	    
    g = pred/(1+pred^2/dlta).*x';
 end




