function D = chi2Kernel(X,Y)
    D = zeros(size(X,1),size(Y,1));
    for i=1:size(Y,1)
        display(i)
        d = bsxfun(@minus, X, Y(i,:));
        s = bsxfun(@plus, X, Y(i,:));
        D(:,i) = sum(d.^2 ./ (s/2+eps), 2);
    end
    D = 1 - D;
end

