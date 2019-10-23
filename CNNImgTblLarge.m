
clc
clear

load('Raw Training data');
load('Y');
test=audio{1};
InputX=size(test,1);
InputY=size(test,2);
% CNN Network topology



layers = [
    imageInputLayer([InputX InputY])
    
    convolution2dLayer(8,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(8,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(2,3,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];





% Training method




options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',50, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');




%Train and test

load('Raw Training data');




Y=categorical(Y);




T=table(audio,Y);



net = trainNetwork(T,layers,options);


save net
