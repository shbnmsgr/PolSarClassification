function C = Classifier(image,class)
C = zeros(size(image,1),size(image,2),max(class));

n = reshape(class,size(image,1),size(image,2));

for i = 1:max(class)
    C(:,:,i) = n == i;  
end

end