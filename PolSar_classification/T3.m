clear,clc,close all
%% Read Features
alpha = mat2gray(imread('Features/alpha.tif'));
anisotropy = mat2gray(imread('Features/anisotropy.tif'));
entropy = mat2gray(imread('Features/entropy.tif'));
combination_1mH1mA = mat2gray(imread('Features/combination_1mH1mA.tif'));
combination_1mHA = mat2gray(imread('Features/combination_1mHA.tif'));
combination_HA = mat2gray(imread('Features/combination_HA.tif'));
Freeman_Dbl_db = mat2gray(imread('Features/Freeman_Dbl_db.tif'));
Freeman_Odd = mat2gray(imread('Features/Freeman_Odd.tif'));
Freeman_Vol = mat2gray(imread('Features/Freeman_Vol.tif'));
Krogager_Kd = mat2gray(imread('Features/Krogager_Kd.tif'));
Krogager_Kh = mat2gray(imread('Features/Krogager_Kh.tif'));
Krogager_Ks = mat2gray(imread('Features/Krogager_Ks.tif'));
span = mat2gray(imread('Features/span.tif'));
TSVM_alpha_s = mat2gray(imread('Features/TSVM_alpha_s.tif'));
TSVM_alpha_s1 = mat2gray(imread('Features/TSVM_alpha_s1.tif'));
TSVM_phi_s = mat2gray(imread('Features/TSVM_phi_s.tif'));
TSVM_psi = mat2gray(imread('Features/TSVM_psi.tif'));
TSVM_tau_m = mat2gray(imread('Features/TSVM_tau_m.tif'));
VanZyl3_Dbl = mat2gray(imread('Features/VanZyl3_Dbl.tif'));
VanZyl3_Odd = mat2gray(imread('Features/VanZyl3_Odd.tif'));
VanZyl3_Vol = mat2gray(imread('Features/VanZyl3_Vol.tif'));

Features = cat(3,alpha,anisotropy,entropy,combination_1mH1mA,combination_1mHA,combination_HA,Freeman_Dbl_db,Freeman_Odd,Freeman_Vol,Krogager_Kd,Krogager_Kh,Krogager_Ks,span,TSVM_alpha_s,TSVM_alpha_s1,TSVM_phi_s,TSVM_psi,TSVM_tau_m,VanZyl3_Dbl,VanZyl3_Odd,VanZyl3_Vol);

Features_1 = Features(:,:,1:7);
Features_2 = Features(:,:,8:14);
Features_3 = Features(:,:,15:21);
Features_4 = Features(:,:,1:14);

Co_occurence_features_4 = imread('Features/Co_occurence_Feature_4_t.tif');
Co_occurence_features_4_d = zeros(size(Co_occurence_features_4));
for i = 1:size(Co_occurence_features_4,3)
    Co_occurence_features_4_d(:,:,i) = mat2gray(Co_occurence_features_4(:,:,i));
end
Features_5 = cat(3,Features_4,Co_occurence_features_4_d);
Features_6 = cat(3,Features_4,Co_occurence_features_4_d(:,:,[1:2:56]));


train = imread('train_6.tif');
test = imread('test_6.tif');

%% A collection of the first 7 features
%% *********************SVM*********************
X = [];
label = [];
for i = 1:6
    ch = [];
    c = train == i;
    n(i) = sum(sum(c));
    for j = 1:size(Features_1,3)
        b = Features_1(:,:,j);
        a = b(c);
        ch(:,j) = a;
    end
    X = [X;ch];
    label = [label;i*ones(n(i),1)];
end
Y = [];
for i =1:size(Features_1,3)
    b = Features_1(:,:,i);
    Y = [Y,b(:)];
end

model = svmtrain(label,X,'-t 3 -s 0 -c 150 -q -b 1');

label = randi([1,6],size(Y,1),1);
[predicted_label, accuracy,prob_estimates] = svmpredict( label,Y, model,'-b 1');

C = Classifier(Features_1,predicted_label');
final = ToRGB(C);
imshow(final),title('first 7 features (SVM)')
[Confiusion_Matrix,overall_ac,user_ac_Features_1,prod_ac_Features_1] = ConfiusionMatrix(test,C);
kappa = KappaCoefficient(Confiusion_Matrix);
figure
names = {'Water','Forest','Winter Wheat','Coniferous','Rye','Urban','User Accuracy'};
for i = 1:6
    subplot(2,3,i)
    imshow(C(:,:,i))
    title(names{i})
end
Water = [Confiusion_Matrix(:,1);100 * user_ac_Features_1(1)]; Forest = [Confiusion_Matrix(:,2);100 * user_ac_Features_1(2)]; Winter_Wheat = [Confiusion_Matrix(:,3);100 * user_ac_Features_1(3)];
Coniferous = [Confiusion_Matrix(:,4);100 * user_ac_Features_1(4)]; Rye = [Confiusion_Matrix(:,5);100 * user_ac_Features_1(5)]; Urban = [Confiusion_Matrix(:,6);100 * user_ac_Features_1(6)];
Producer_Accuracy = 100*[prod_ac_Features_1;overall_ac];
table(Water,Forest,Winter_Wheat,Coniferous,Rye,Urban,Producer_Accuracy,'rownames',names)
%% *********************K-Means*********************
K1 = 6;
M =zeros(size(Features_1,3),K1);
while true
    % Random Centers
    for z =1:size(Features_1,3)
        M(z,:) = randi([0,255],1,K1);
    end
    M = mat2gray(M);
    
    SSE1 = [];
    dSSE1 = [];
    for i = 1:100
        
        Dist1 = EuclideanDistCompute(Features_1,M);
        
        C1 = Classifier_Kmeans(Features_1,Dist1);
        
        M = MeanComputer(Features_1,C1);
        
        SSE1(i) = sum(sum(Dist1.^2));
        if i>1
            dSSE1(i-1) = abs(SSE1(i)- SSE1(i-1));
            if dSSE1(i-1)<1e-20
                break
            end
        end
        
    end
    
    if i<100
        break
    end
end

final1 = ToRGB(C1);
imshow(final1),title('first 7 features (K-Means)')
[TP,TN,FP,FN,RI,JI] = RandIndex(C1,test);
[Confiusion_Matrix,overall_ac,user_ac_Features_1,prod_ac_Features_1] = ConfiusionMatrix(test,C1);
figure
names = {'Water','Forest','Winter Wheat','Coniferous','Rye','Urban','User Accuracy'};
for i = 1:6
    subplot(2,3,i)
    imshow(C1(:,:,i))
    title(names{i})
end


%% A collection of the second 7 features
%% *********************SVM*********************
X = [];
label = [];
for i = 1:6
    ch = [];
    c = train == i;
    n(i) = sum(sum(c));
    for j = 1:size(Features_2,3)
        b = Features_2(:,:,j);
        a = b(c);
        ch(:,j) = a;
    end
    X = [X;ch];
    label = [label;i*ones(n(i),1)];
end
Y = [];
for i =1:size(Features_2,3)
    b = Features_2(:,:,i);
    Y = [Y,b(:)];
end

model = svmtrain(label,X,'-t 3 -s 0 -c 150 -q -b 1');

label = randi([1,6],size(Y,1),1);
[predicted_label, accuracy,prob_estimates] = svmpredict( label,Y, model,'-b 1');

C = Classifier(Features_2,predicted_label');
final = ToRGB(C);
imshow(final),title('second 7 features (SVM)')
[Confiusion_Matrix,overall_ac,user_ac_Features_2,prod_ac_Features_2] = ConfiusionMatrix(test,C);
kappa = KappaCoefficient(Confiusion_Matrix)
figure
names = {'Water','Forest','Winter Wheat','Coniferous','Rye','Urban','User Accuracy'};
for i = 1:6
    subplot(2,3,i)
    imshow(C(:,:,i))
    title(names{i})
end
Water = [Confiusion_Matrix(:,1);100 * user_ac_Features_2(1)]; Forest = [Confiusion_Matrix(:,2);100 * user_ac_Features_2(2)]; Winter_Wheat = [Confiusion_Matrix(:,3);100 * user_ac_Features_2(3)];
Coniferous = [Confiusion_Matrix(:,4);100 * user_ac_Features_2(4)]; Rye = [Confiusion_Matrix(:,5);100 * user_ac_Features_2(5)]; Urban = [Confiusion_Matrix(:,6);100 * user_ac_Features_2(6)];
Producer_Accuracy = 100*[prod_ac_Features_2;overall_ac];
table(Water,Forest,Winter_Wheat,Coniferous,Rye,Urban,Producer_Accuracy,'rownames',names)

%% *********************K-Means*********************
K1 = 6;
M =zeros(size(Features_2,3),K1);
while true
    %     Random Centers
    for z =1:size(Features_2,3)
        M(z,:) = randi([0,255],1,K1);
    end
    M = mat2gray(M);
    
    
    SSE1 = [];
    dSSE1 = [];
    for i = 1:100
        
        Dist1 = EuclideanDistCompute(Features_2,M);
        
        C1 = Classifier_Kmeans(Features_2,Dist1);
        
        M = MeanComputer(Features_2,C1);
        
        SSE1(i) = sum(sum(Dist1.^2));
        if i>1
            dSSE1(i-1) = abs(SSE1(i)- SSE1(i-1));
            if dSSE1(i-1)<1e-20
                break
            end
        end
        
    end
    
    if i<100
        break
    end
end

final1 = ToRGB(C1);
imshow(final1),title('second 7 features (K-Means)')
[TP,TN,FP,FN,RI,JI] = RandIndex(C1,test);
[Confiusion_Matrix,overall_ac,user_ac_Features_2,prod_ac_Features_2] = ConfiusionMatrix(test,C1);
figure
names = {'Water','Forest','Winter Wheat','Coniferous','Rye','Urban','User Accuracy'};
for i = 1:6
    subplot(2,3,i)
    imshow(C1(:,:,i))
    title(names{i})
end


%% A collection of the third 7 features
%% *********************SVM*********************
X = [];
label = [];
for i = 1:6
    ch = [];
    c = train == i;
    n(i) = sum(sum(c));
    for j = 1:size(Features_3,3)
        b = Features_3(:,:,j);
        a = b(c);
        ch(:,j) = a;
    end
    X = [X;ch];
    label = [label;i*ones(n(i),1)];
end
Y = [];
for i =1:size(Features_3,3)
    b = Features_3(:,:,i);
    Y = [Y,b(:)];
end

model = svmtrain(label,X,'-t 3 -s 0 -c 150 -q -b 1');

label = randi([1,6],size(Y,1),1);
[predicted_label, accuracy,prob_estimates] = svmpredict( label,Y, model,'-b 1');

C = Classifier(Features_3,predicted_label');
final = ToRGB(C);
imshow(final),title('third 7 features (SVM)')
[Confiusion_Matrix,overall_ac,user_ac_Features_3,prod_ac_Features_3] = ConfiusionMatrix(test,C);
kappa = KappaCoefficient(Confiusion_Matrix)
figure
names = {'Water','Forest','Winter Wheat','Coniferous','Rye','Urban','User Accuracy'};
for i = 1:6
    subplot(2,3,i)
    imshow(C(:,:,i))
    title(names{i})
end
Water = [Confiusion_Matrix(:,1);100 * user_ac_Features_3(1)]; Forest = [Confiusion_Matrix(:,2);100 * user_ac_Features_3(2)]; Winter_Wheat = [Confiusion_Matrix(:,3);100 * user_ac_Features_3(3)];
Coniferous = [Confiusion_Matrix(:,4);100 * user_ac_Features_3(4)]; Rye = [Confiusion_Matrix(:,5);100 * user_ac_Features_3(5)]; Urban = [Confiusion_Matrix(:,6);100 * user_ac_Features_3(6)];
Producer_Accuracy = 100*[prod_ac_Features_3;overall_ac];
table(Water,Forest,Winter_Wheat,Coniferous,Rye,Urban,Producer_Accuracy,'rownames',names)

%% *********************K-Means*********************
K1 = 6;
M =zeros(size(Features_3,3),K1);
while true
    % Random Centers
    for z =1:size(Features_3,3)
        M(z,:) = randi([0,255],1,K1);
    end
    M = mat2gray(M);
    
    
    SSE1 = [];
    dSSE1 = [];
    for i = 1:100
        i
        Dist1 = EuclideanDistCompute(Features_3,M);
        
        C1 = Classifier_Kmeans(Features_3,Dist1);
        
        M = MeanComputer(Features_3,C1);
        
        SSE1(i) = sum(sum(Dist1.^2));
        if i>1
            dSSE1(i-1) = abs(SSE1(i)- SSE1(i-1));
            if dSSE1(i-1)<1e-20
                break
            end
        end
        
    end
    
    if i<100
        break
    end
end

final1 = ToRGB(C1);
imshow(final1),title('third 7 features (K-Means)')
[TP,TN,FP,FN,RI,JI] = RandIndex(C1,test);
[Confiusion_Matrix,overall_ac,user_ac_Features_3,prod_ac_Features_3] = ConfiusionMatrix(test,C1);
figure
names = {'Water','Forest','Winter Wheat','Coniferous','Rye','Urban','User Accuracy'};
for i = 1:6
    subplot(2,3,i)
    imshow(C1(:,:,i))
    title(names{i})
end


%% A collection of the first 14 features
%% *********************SVM*********************
X = [];
label = [];
for i = 1:6
    ch = [];
    c = train == i;
    n(i) = sum(sum(c));
    for j = 1:size(Features_4,3)
        b = Features_4(:,:,j);
        a = b(c);
        ch(:,j) = a;
    end
    X = [X;ch];
    label = [label;i*ones(n(i),1)];
end
Y = [];
for i =1:size(Features_4,3)
    b = Features_4(:,:,i);
    Y = [Y,b(:)];
end

model = svmtrain(label,X,'-t 3 -s 0 -c 150 -q -b 1');

label = randi([1,6],size(Y,1),1);
[predicted_label, accuracy,prob_estimates] = svmpredict( label,Y, model,'-b 1');

C = Classifier(Features_4,predicted_label');
final = ToRGB(C);
imshow(final),title('first 14 features (SVM)')
[Confiusion_Matrix,overall_ac,user_ac_Features_4,prod_ac_Features_4] = ConfiusionMatrix(test,C);
kappa = KappaCoefficient(Confiusion_Matrix)
figure
names = {'Water','Forest','Winter Wheat','Coniferous','Rye','Urban','User Accuracy'};
for i = 1:6
    subplot(2,3,i)
    imshow(C(:,:,i))
    title(names{i})
end
Water = [Confiusion_Matrix(:,1);100 * user_ac_Features_4(1)]; Forest = [Confiusion_Matrix(:,2);100 * user_ac_Features_4(2)]; Winter_Wheat = [Confiusion_Matrix(:,3);100 * user_ac_Features_4(3)];
Coniferous = [Confiusion_Matrix(:,4);100 * user_ac_Features_4(4)]; Rye = [Confiusion_Matrix(:,5);100 * user_ac_Features_4(5)]; Urban = [Confiusion_Matrix(:,6);100 * user_ac_Features_4(6)];
Producer_Accuracy = 100*[prod_ac_Features_4;overall_ac];
table(Water,Forest,Winter_Wheat,Coniferous,Rye,Urban,Producer_Accuracy,'rownames',names)

%% *********************K-Means*********************
K1 = 6;
M =zeros(size(Features_4,3),K1);
while true
    %     Random Centers
    for z =1:size(Features_4,3)
        M(z,:) = randi([0,255],1,K1);
    end
    M = mat2gray(M);
    
    
    SSE1 = [];
    dSSE1 = [];
    for i = 1:150
        i
        Dist1 = EuclideanDistCompute(Features_4,M);
        
        C1 = Classifier_Kmeans(Features_4,Dist1);
        
        M = MeanComputer(Features_4,C1);
        
        SSE1(i) = sum(sum(Dist1.^2));
        if i>1
            dSSE1(i-1) = abs(SSE1(i)- SSE1(i-1));
            if dSSE1(i-1)<1e-20
                break
            end
        end
        
    end
    
    if i<150
        break
    end
end

final1 = ToRGB(C1);
imshow(final1),title('first 14 features (K-Means)')
[TP,TN,FP,FN,RI,JI] = RandIndex(C1,test);
[Confiusion_Matrix,overall_ac,user_ac_Features_4,prod_ac_Features_4] = ConfiusionMatrix(test,C1);
figure
names = {'Water','Forest','Winter Wheat','Coniferous','Rye','Urban','User Accuracy'};
for i = 1:6
    subplot(2,3,i)
    imshow(C1(:,:,i))
    title(names{i})
end




% *****************************************************************************************************************
% *****************************************************************************************************************
%% 5-2
%% all Co-occurence textures + Features_4
%% *********************SVM*********************
X = [];
label = [];
for i = 1:6
    ch = [];
    c = train == i;
    n(i) = sum(sum(c));
    for j = 1:size(Features_5,3)
        b = Features_5(:,:,j);
        a = b(c);
        ch(:,j) = a;
    end
    X = [X;ch];
    label = [label;i*ones(n(i),1)];
end
Y = [];
for i =1:size(Features_5,3)
    b = Features_5(:,:,i);
    Y = [Y,b(:)];
end

model = svmtrain(label,X,'-t 3 -s 0 -c 150 -q -b 1');

label = randi([1,6],size(Y,1),1);
[predicted_label, accuracy,prob_estimates] = svmpredict( label,Y, model,'-b 1');

C = Classifier(Features_5,predicted_label');
final = ToRGB(C);
imshow(final),title('first 14 features + Texture Features (SVM)')
[Confiusion_Matrix,overall_ac,user_ac_Features_5,prod_ac_Features_5] = ConfiusionMatrix(test,C);
kappa = KappaCoefficient(Confiusion_Matrix)
figure
names = {'Water','Forest','Winter Wheat','Coniferous','Rye','Urban','User Accuracy'};
for i = 1:6
    subplot(2,3,i)
    imshow(C(:,:,i))
    title(names{i})
end
Water = [Confiusion_Matrix(:,1);100 * user_ac_Features_5(1)]; Forest = [Confiusion_Matrix(:,2);100 * user_ac_Features_5(2)]; Winter_Wheat = [Confiusion_Matrix(:,3);100 * user_ac_Features_5(3)];
Coniferous = [Confiusion_Matrix(:,4);100 * user_ac_Features_5(4)]; Rye = [Confiusion_Matrix(:,5);100 * user_ac_Features_5(5)]; Urban = [Confiusion_Matrix(:,6);100 * user_ac_Features_5(6)];
Producer_Accuracy = 100*[prod_ac_Features_5;overall_ac];
table(Water,Forest,Winter_Wheat,Coniferous,Rye,Urban,Producer_Accuracy,'rownames',names)

%% *********************K-Means*********************
K1 = 6;
M =zeros(size(Features_5,3),K1);
while true
    %     Random Centers
%     for z =1:size(Features_5,3)
%         M(z,:) = randi([0,255],1,K1);
%     end
%     M = mat2gray(M);
%     
 
for i = 1:size(Features_5,3)
    b = Features_5(:,:,i);
    for j = 1:K1
        M(i,j) = mean(b(train == j));
    end
end

    SSE1 = [];
    dSSE1 = [];
    for i = 1:150
        i
        Dist1 = EuclideanDistCompute(Features_5,M);
        
        C1 = Classifier_Kmeans(Features_5,Dist1);
        
        M = MeanComputer(Features_5,C1);
        
        SSE1(i) = sum(sum(Dist1.^2));
        if i>1
            dSSE1(i-1) = abs(SSE1(i)- SSE1(i-1));
            if dSSE1(i-1)<1e-20
                break
            end
        end
        
    end
    
    if i<150
        break
    end
end

final1 = ToRGB(C1);
imshow(final1),title('first 14 features + Texture Features(K-Means)')
[TP,TN,FP,FN,RI,JI] = RandIndex(C1,test);
[Confiusion_Matrix,overall_ac,user_ac_Features_5,prod_ac_Features_5] = ConfiusionMatrix(test,C1);
figure
names = {'Water','Forest','Winter Wheat','Coniferous','Rye','Urban','User Accuracy'};
for i = 1:6
    subplot(2,3,i)
    imshow(C1(:,:,i))
    title(names{i})
end
