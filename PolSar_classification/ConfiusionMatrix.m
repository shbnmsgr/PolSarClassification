function [Confiusion_Matrix,overall_ac,user_ac,prod_ac] = ConfiusionMatrix(test,C)
Confiusion_Matrix = zeros(size(C,3));
testval = unique(test(:));
testval = testval(2:end);
TC = logical(zeros(size(C)));

for i = 1:size(C,3)
    n = testval(i);
    ch = test == n;
    TC(:,:,i) = ch;
end

for i = 1:size(C,3)
    tc = TC(:,:,i);
    for j = 1:size(C,3)
        c = C(:,:,j);
        Confiusion_Matrix(i,j) = sum(sum(c(tc)));
    end
end

overall_ac = trace(Confiusion_Matrix)/sum(sum(Confiusion_Matrix));
user_ac = diag(Confiusion_Matrix)./sum(Confiusion_Matrix)';
prod_ac = diag(Confiusion_Matrix)./sum(Confiusion_Matrix,2);
end