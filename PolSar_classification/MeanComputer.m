function M = MeanComputer(image5,C)

for i =1:size(image5,3)
    b = image5(:,:,i);
    for j = 1:size(C,3)
        c = b(C(:,:,j) == 1);
        M(i,j) = mean(mean(c));
    end
end

end