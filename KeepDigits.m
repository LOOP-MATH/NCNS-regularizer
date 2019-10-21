function w = KeepDigits(w,p)

% keep p digits of vector w;  
ind = find(w);
u = w(ind);
digit=ceil(log10(abs(u)));
u = round(u./(10.^(digit-p))).*10.^(digit-p);
w(ind) = u;

