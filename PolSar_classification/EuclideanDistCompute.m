function Dist = EuclideanDistCompute(image5,M)
% M is Nfeatures * Nclasses size
% X is Nsamples * Nfeatures size
% Dist is Nsamples * Nclasses size
X = zeros(size(image5,1) * size(image5,2),size(image5,3));
for i =1:size(image5,3)
    b = image5(:,:,i);
    X(:,i) = b(:);
end

Dist = zeros(size(X,1),size(M,2));
for i =1:size(M,2)
    s = 0;
    for j = 1:size(X,2)
        s = s + (X(:,j) - M(j,i)).^2;
    end
    Dist(:,i) = sqrt(s);
end
end