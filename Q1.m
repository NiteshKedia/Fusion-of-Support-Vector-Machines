load('TestData.mat');
load('TrainData.mat');
%Step 0
%1)
H1 = svmtrain(YTrain,X1Train, '-c 4 -t 0');
H2 = svmtrain(YTrain,X2Train, '-c 4 -t 0');
H3 = svmtrain(YTrain,X3Train, '-c 4 -t 0');
[predict_label1, accuracy1, dec_values1] = svmpredict(YTest, X1Test, H1);
[predict_label2, accuracy2, dec_values2] = svmpredict(YTest, X2Test, H2);
[predict_label3, accuracy3, dec_values3] = svmpredict(YTest, X3Test, H3);

%2)
H1PP = svmtrain(YTrain,X1Train, '-b 1 -c 4 -t 0');
H2PP = svmtrain(YTrain,X2Train, '-b 1 -c 4 -t 0');
H3PP = svmtrain(YTrain,X3Train, '-b 1 -c 4 -t 0');
[predict_label1PP, accuracy1PP, dec_values1PP] = svmpredict(YTest, X1Test, H1PP, '-b 1');
[predict_label2PP, accuracy2PP, dec_values2PP] = svmpredict(YTest, X2Test, H2PP,'-b 1');
[predict_label3PP, accuracy3PP, dec_values3PP] = svmpredict(YTest, X3Test, H3PP,'-b 1');

%Step 1
%1)
TestLabelStep2=[];
Accuracy_count_Step1=0;
pwix=(dec_values1PP +  dec_values2PP + dec_values3PP)/3;
for i=1:1883
    [row,col] = find(pwix == max(pwix(i,:)),1,'first');
    TestLabelStep2 = [TestLabelStep2;col];
    if col == YTest(i)
        Accuracy_count_Step1=Accuracy_count_Step1+1;
    end
end
Accuracy_Step1 = Accuracy_count_Step1*100/1883;

%2)

X_Train= [X1Train , X2Train , X3Train];
X_Test= [X1Test , X2Test , X3Test];

HStep2 = svmtrain(YTrain,X_Train, '-c 4 -t 0');
[predict_label_Step2, accuracy_Step2, dec_values_Step2] = svmpredict(YTest, X_Test, HStep2);

%Step3
%1
K1Train =  [ (1:4786)' , chi2Kernel(X1Train,X1Train) ];
K2Train =  [ (1:4786)' , chi2Kernel(X2Train,X2Train) ];
K3Train =  [ (1:4786)' , chi2Kernel(X3Train,X3Train) ];

H1Step3 = svmtrain(YTrain, K1Train, '-c 4 -t 4');
H2Step3 = svmtrain(YTrain, K2Train, '-c 4 -t 4');
H3Step3 = svmtrain(YTrain, K3Train, '-c 4 -t 4');

K1Test = [ (1:1883)'  , chi2Kernel(X1Test,X1Train)  ];
K2Test = [ (1:1883)'  , chi2Kernel(X2Test,X2Train)  ];
K3Test = [ (1:1883)'  , chi2Kernel(X3Test,X3Train)  ];

[predict_label1Step3, accuracy1Step3, dec_values1Step3] = svmpredict(YTest, K1Test, H1Step3);
[predict_label2Step3, accuracy2Step3, dec_values2Step3] = svmpredict(YTest, K2Test, H2Step3);
[predict_label3Step3, accuracy3Step3, dec_values3Step3] = svmpredict(YTest, K3Test, H3Step3);

%2


% i)
KaTrain=(K1Train+K2Train+K3Train)/3;
KaTest=(K1Test+K2Test+K3Test)/3;
HKaStep3= svmtrain(YTrain,KaTrain, '-c 4 -t 4');
[predict_labelKaStep3, accuracyKaStep3, dec_valuesKaStep3] = svmpredict(YTest, KaTest, HKaStep3);

% ii)
K1Train_bak= K1Train;
K2Train_bak= K2Train;
K3Train_bak= K3Train;
K1Train(:,1) = [];
K2Train(:,1) = [];
K3Train(:,1) = [];

K1Test_bak= K1Test;
K2Test_bak= K2Test;
K3Test_bak= K3Test;
K1Test(:,1) = [];
K2Test(:,1) = [];
K3Test(:,1) = [];



KbTraintemp = (K1Train.*K2Train.*K3Train).^(1/3);
KbTesttemp =  (K1Test.*K2Test.*K3Test).^(1/3);
KbTraintemp=abs(KbTraintemp);
KbTesttemp=abs(KbTesttemp);

KbTraintemp= [(1:4786)'  ,KbTraintemp];
KbTesttemp=  [(1:1883)'  ,KbTesttemp];

HKbStep3= svmtrain(YTrain,KbTrain, '-c 4 -t 4');
[predict_labelKbStep3, accuracyKbStep3, dec_valuesKbStep3] = svmpredict(YTest, KbTest, HKbStep3);




% mn = min(K1Train,[],1); mx = max(K1Train,[],1);
% K1Train = bsxfun(@rdivide, bsxfun(@minus, K1Train, mn), mx-mn);
% mn = min(K2Train,[],1); mx = max(K2Train,[],1);
% K2Train = bsxfun(@rdivide, bsxfun(@minus, K2Train, mn), mx-mn);
% mn = min(K3Train,[],1); mx = max(K3Train,[],1);
% K3Train = bsxfun(@rdivide, bsxfun(@minus, K3Train, mn), mx-mn);
% mn = min(K1Test,[],1); mx = max(K1Test,[],1);
% K1Test = bsxfun(@rdivide, bsxfun(@minus, K1Test, mn), mx-mn);
% mn = min(K2Test,[],1); mx = max(K2Test,[],1);
% K2Test = bsxfun(@rdivide, bsxfun(@minus, K2Test, mn), mx-mn);
% mn = min(K3Test,[],1); mx = max(K3Test,[],1);
% K3Test = bsxfun(@rdivide, bsxfun(@minus, K3Test, mn), mx-mn);
% 
% 
% 
% 
% KbTrain = [ (1:4786)'  , (K1Train.*K2Train.*K3Train).^(1/3);  ];
% KbTest = [ (1:1883)'  , (K1Test.*K2Test.*K3Test).^(1/3);  ];
% 
% 
% HKbStep3= svmtrain(YTrain,KbTrain, '-c 4 -t 4');
% [predict_labelKbStep3, accuracyKbStep3, dec_valuesKbStep3] = svmpredict(YTest, KbTest, HKbStep3);





