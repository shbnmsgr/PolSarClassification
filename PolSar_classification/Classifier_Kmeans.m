function C = Classifier_Kmeans(image5,Dist)
C = zeros(size(image5,1),size(image5,2),size(Dist,2));


[~,n] = min(Dist');
n = n';

n = reshape(n,size(image5,1),size(image5,2));

for i = 1:size(Dist,2)
    C(:,:,i) = n == i;
end

end