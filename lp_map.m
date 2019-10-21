function v = lp_map(v,lambda,p)

if p < 0 % l0 constraint: ||v||_0 \leq \lambda
   [val ind] = sort(abs(v),'descend');
   index = ind(ceil(lambda)+1:end);
   v(index) = 0;
end


if p == 1
  % soft thresholding
  % min_x  1/2 ||x - v||_2^2 +  lambda ||x||_1  
  v = sign(v).*max(abs(v)-lambda,0);
end

if p == 0
  % hard-thresholding 
  % min_x  1/2 ||x - v||_2^2 +  lambda ||x||_0  
  v(abs(v)<=sqrt(2*lambda)) = 0; 
end


if p == .5
   % min_x  1/2 ||x - v||_2^2 +  lambda ||x||_{1/2} 
   u = abs(v);
   id = find(u>1.5*lambda^(2/3));
   temp = 1 + cos(2/3*(pi-acos(0.25*lambda.*((u(id)/3).^(-1.5)))));
   temp = (2/3).*(v(id).*temp);
   [a,b]=size(v);
   v = zeros(a,b);
   v(id) = temp;
end
