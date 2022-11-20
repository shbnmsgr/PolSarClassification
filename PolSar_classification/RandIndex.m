function [TP,TN,FP,FN,RI,JI] = RandIndex(C,test)

t = [double(test(:)),[1:length(test(:))]'];
M = find(t(:,1) ~= 0);

ch = nchoosek([1:6000],2);


b1 = zeros(size(test));
b1(C(:,:,1) == 1) = 1;
b1(C(:,:,2) == 1) = 2;
b1(C(:,:,3) == 1) = 3;
b1(C(:,:,4) == 1) = 4;
b1(C(:,:,5) == 1) = 5;
b1(C(:,:,6) == 1) = 6;

TP = 0; TN = 0; FP = 0; FN = 0;
for i =1:length(ch)
    
    s1 = ch(i,1); s2 = ch(i,2);
    if t(M(s1),1) == t(M(s2),1) %same class
        if b1(M(s1)) == b1(M(s2)) %same cluster
            TP = TP + 1;
        elseif b1(M(s1)) ~= b1(M(s2)) %different cluster
            FN = FN + 1;
        end
    elseif t(M(s1)) ~= t(M(s2)) %different class
        if b1(M(s1)) == b1(M(s2)) %same cluster
            FP = FP + 1;
        elseif b1(M(s1)) ~= b1(M(s2)) %different cluster
            TN = TN + 1;
        end
        
    end
    
    

    
end

RI = (TP + TN)/(TP + TN + FP + FN);
JI = (TP)/(TP + FP + FN);

name = {'Same Class','Different Class'};

Same_Cluster = [TP;FP];
Different_Cluster = [FN;TN];
table(Same_Cluster,Different_Cluster,'rownames',name)

end