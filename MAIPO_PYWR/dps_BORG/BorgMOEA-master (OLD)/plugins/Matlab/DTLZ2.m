function [ objs ] = DTLZ2( x )
    nvars = 11;
    nobjs = 2;

    g = sum((x(nobjs:nvars) - 0.5*ones(1, nvars-nobjs+1)).^2);
    objs = (1+g)*ones(1, nobjs);
    
    for i=1:nobjs
        for j=1:nobjs-i
            objs(i) = objs(i) * cos(0.5 * pi * x(j));
        end
        
        if i > 1
            objs(i) = objs(i) * sin(0.5 * pi * x(nobjs-i+1));
        end
    end

end

