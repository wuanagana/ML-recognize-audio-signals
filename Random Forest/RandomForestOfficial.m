clc; clear;
%% Acquisition of train and test signal and noise

% Train noise
trainN_ds_frame = mirframe('Ufficiali/trainRumore.wav', 'Length', 1024, 'sp', 'Hop', 256, 'sp');
trainN_ds = mirspectrum(trainN_ds_frame,'Window','hamming','dB');
% Test Noise
testN_ds_frame = mirframe('Ufficiali/testRumore.wav','Length',1024,'sp','Hop',256,'sp');
testN_ds = mirspectrum(testN_ds_frame,'Window','hamming','dB');

% Train signal
trainS_ds_frame = mirframe('Ufficiali/trainSuono.wav', 'Length', 1024, 'sp', 'Hop', 256, 'sp');
trainS_ds = mirspectrum(trainS_ds_frame,'Window','hamming','dB');
% Test signal
testS_ds_frame = mirframe('Ufficiali/testSuonoBeam.wav','Length',1024,'sp','Hop',256,'sp');
testS_ds = mirspectrum(testS_ds_frame,'Window','hamming','dB');

%% Feature extraction
%In this section the features are extracted. These features come from MIRToolbox.

% RMS energy
rms_ds_testN = mirgetdata(mirrms(testN_ds_frame));
rms_ds_testS = mirgetdata(mirrms(testS_ds_frame));
rms_ds_trainN = mirgetdata(mirrms(trainN_ds_frame));
rms_ds_trainS = mirgetdata(mirrms(trainS_ds_frame));

% Flux
flux_ds_testN = mirgetdata(mirflux(testN_ds));
flux_ds_testS = mirgetdata(mirflux(testS_ds));
flux_ds_trainN = mirgetdata(mirflux(trainN_ds));
flux_ds_trainS = mirgetdata(mirflux(trainS_ds));

% Centroid
centroid_ds_testN = mirgetdata(mircentroid(testN_ds)); 
centroid_ds_testS = mirgetdata(mircentroid(testS_ds));
centroid_ds_trainN = mirgetdata(mircentroid(trainN_ds));
centroid_ds_trainS = mirgetdata(mircentroid(trainS_ds));

% Low energy
lenergy_ds_testN = mirgetdata(mirlowenergy(testN_ds));
lenergy_ds_testS = mirgetdata(mirlowenergy(testS_ds));
lenergy_ds_trainN = mirgetdata(mirlowenergy(trainN_ds));
lenergy_ds_trainS = mirgetdata(mirlowenergy(trainS_ds));

% Spread
spread_ds_trainN = mirgetdata(mirspread(trainN_ds));
spread_ds_trainS = mirgetdata(mirspread(trainS_ds));
spread_ds_testN = mirgetdata(mirspread(testN_ds));
spread_ds_testS = mirgetdata(mirspread(testS_ds));

% zero-cross

zcross_ds_trainN = mirgetdata(mirzerocross(trainN_ds));
zcross_ds_trainS = mirgetdata(mirzerocross(trainS_ds));
zcross_ds_testN = mirgetdata(mirzerocross(testN_ds));
zcross_ds_testS = mirgetdata(mirzerocross(testS_ds));

% roll-off

roff_ds_trainN = mirgetdata(mirrolloff(trainN_ds));
roff_ds_trainS = mirgetdata(mirrolloff(trainS_ds));
roff_ds_testN = mirgetdata(mirrolloff(testN_ds));
roff_ds_testS = mirgetdata(mirrolloff(testS_ds));

% brightness

bri_ds_trainN = mirgetdata(mirbrightness(trainN_ds));
bri_ds_trainS = mirgetdata(mirbrightness(trainS_ds));
bri_ds_testN = mirgetdata(mirbrightness(testN_ds));
bri_ds_testS = mirgetdata(mirbrightness(testS_ds));

% kurtosis
kurt_ds_trainN = mirgetdata(mirkurtosis(trainN_ds));
kurt_ds_trainS = mirgetdata(mirkurtosis(trainS_ds));
kurt_ds_testN = mirgetdata(mirkurtosis(testN_ds));
kurt_ds_testS = mirgetdata(mirkurtosis(testS_ds));

%% Regolarization of features 
% in this section the length of some samples are cut in order
% to have all the features with the same size.

centroid_ds_trainN = centroid_ds_trainN(1:1870);
centroid_ds_testN = centroid_ds_testN(1:1870);

flux_ds_trainN = flux_ds_trainN(1:1870);
flux_ds_testN = flux_ds_testN(1:1870);

rms_ds_trainN = rms_ds_trainN(1:1870);
rms_ds_testN = rms_ds_testN(1:1870);

centroid_ds_trainS = centroid_ds_trainS(1:1870);
centroid_ds_testS = centroid_ds_testS(1:1870);

flux_ds_trainS = flux_ds_trainS(1:1870);
flux_ds_testS = flux_ds_testS(1:1870);

rms_ds_trainS = rms_ds_trainS(1:1870);
rms_ds_testS = rms_ds_testS(1:1870);

lenergy_ds_testN = lenergy_ds_testN(1:1870);
lenergy_ds_testS = lenergy_ds_testS(1:1870);

lenergy_ds_trainN = lenergy_ds_trainN(1:1870);
lenergy_ds_trainS = lenergy_ds_trainS(1:1870);

spread_ds_trainN = spread_ds_trainN(1:1870);
spread_ds_trainS = spread_ds_trainS(1:1870);

spread_ds_testN = spread_ds_testN(1:1870);
spread_ds_testS = spread_ds_testS(1:1870);

zcross_ds_trainN = zcross_ds_trainN(1:1870);
zcross_ds_trainS = zcross_ds_trainS(1:1870);

zcross_ds_testN = zcross_ds_testN(1:1870);
zcross_ds_testS = zcross_ds_testS(1:1870);

roff_ds_trainN = roff_ds_trainN(1:1870);
roff_ds_trainS = roff_ds_trainS(1:1870);

roff_ds_testN = roff_ds_testN(1:1870);
roff_ds_testS = roff_ds_testS(1:1870);

bri_ds_trainN = bri_ds_trainN(1:1870);
bri_ds_trainS = bri_ds_trainS(1:1870);

bri_ds_testN = bri_ds_testN(1:1870);
bri_ds_testS = bri_ds_testS(1:1870);

kurt_ds_trainN = kurt_ds_trainN(1:1870);
kurt_ds_trainS = kurt_ds_trainS(1:1870);

kurt_ds_testN = kurt_ds_testN(1:1870);
kurt_ds_testS = kurt_ds_testS(1:1870);

%% Building input features for training model on D&S signal
% in this section we manipulate the feature arrays to make them compatible 
% to the training model input. Then the input model for the Random Forest Algorithm
% is created.

rms_ds_train = vertcat(rms_ds_trainS',rms_ds_trainN');
flux_ds_train = vertcat(flux_ds_trainS',flux_ds_trainN');
centroid_ds_train = vertcat(centroid_ds_trainS',centroid_ds_trainN');
lenergy_ds_train = vertcat(lenergy_ds_trainS',lenergy_ds_trainN');
spread_ds_train = vertcat(spread_ds_trainS',spread_ds_trainN');
zcross_ds_train = vertcat(zcross_ds_trainS',zcross_ds_trainN');
roff_ds_train = vertcat(roff_ds_trainS',roff_ds_trainN');
bri_ds_train = vertcat(bri_ds_trainS',bri_ds_trainN');
kurt_ds_train = vertcat(kurt_ds_trainS',kurt_ds_trainN');

% save('training_feat.mat','rms_ds_train','flux_ds_train','centroid_ds_train');
% load('training_feat.mat');

for i = 1:size(rms_ds_trainS,2)
    type(i) = "sound";
end
for i = (size(rms_ds_trainS,2)+1):(size(rms_ds_trainS,2)+size(rms_ds_trainN,2))
    type(i) = "noise";
end

cellstype = cellstr(type');
rmsT = array2table(rms_ds_train);
fluxT = array2table(flux_ds_train);
centroidT = array2table(centroid_ds_train);
lenergyT = array2table(lenergy_ds_train);
spreadT = array2table(spread_ds_train);
zcrossT = array2table(zcross_ds_train);
roffT = array2table(roff_ds_train);
briT = array2table(bri_ds_train);
kurtT = array2table(kurt_ds_train);


input = [rmsT fluxT centroidT lenergyT spreadT zcrossT roffT briT kurtT cellstype];
input.Properties.VariableNames{10} = 'Type';

%% Creation of the model with optimal parameters
% After test every single case (NumTrees, learRate, maxNumSplits), the 
% optimal parameters are established.
% The optimal model is created with these parameters.
 
NumTrees = [50, 100, 150];
n = size(input,1);
m = floor(log(n - 1)/log(3));
learnRate = [0.1 0.25 0.5 1];
maxNumSplits = 3.^(0:m);

load('Usefull_data.mat');
tFinal = templateTree('MaxNumSplits',maxNumSplits(idxMNS500));
MdlFinal = fitcensemble(input,'Type','NumLearningCycles',NumTrees(3),'Learners',tFinal,'LearnRate',learnRate(idxLR500));


%% Building inputs for testing the model on the DS signal
% in this section we manipulate the feature arrays to make them compatible 
% to the testing model input. Then the input model for the Random Forest Algorithm
% is created.

rms_ds_test = vertcat(rms_ds_testS',rms_ds_testN');
flux_ds_test = vertcat(flux_ds_testS',flux_ds_testN');
centroid_ds_test = vertcat(centroid_ds_testS',centroid_ds_testN');
lenergy_ds_test = vertcat(lenergy_ds_testS',lenergy_ds_testN');
spread_ds_test = vertcat(spread_ds_testS',spread_ds_testN');
zcross_ds_test = vertcat(zcross_ds_testS',zcross_ds_testN');
roff_ds_test = vertcat(roff_ds_testS',roff_ds_testN');
bri_ds_test = vertcat(bri_ds_testS',bri_ds_testN');
kurt_ds_test = vertcat(kurt_ds_testS',kurt_ds_testN');

for i = 1:(size(rms_ds_testS,2))
    type_test(i) = "sound";
end
for i = ((size(rms_ds_testS,2))+1):(size(rms_ds_testS,2)+size(rms_ds_testN,2))
    type_test(i) = "noise";
end
rms_ds_train = rms_ds_test;
flux_ds_train = flux_ds_test;
centroid_ds_train = centroid_ds_test;
lenergy_ds_train = lenergy_ds_test;
spread_ds_train = spread_ds_test;
zcross_ds_train = zcross_ds_test;
roff_ds_train = roff_ds_test;
bri_ds_train = bri_ds_test;
% entr_ds_train = entr_ds_test;
kurt_ds_train = kurt_ds_test;

cellstype = cellstr(type_test');
rms = array2table(rms_ds_train);
flux = array2table(flux_ds_train);
centroid = array2table(centroid_ds_train);
lenergy = array2table(lenergy_ds_train);
spread = array2table(spread_ds_train);
zcross = array2table(zcross_ds_train);
roff = array2table(roff_ds_train);
bri = array2table(bri_ds_train);
% entr = array2table(entr_ds_train);
kurt = array2table(kurt_ds_train);

input_test = [rms flux centroid lenergy spread zcross roff bri kurt];

%% Predictions of signal
% in this section they are predict the samples with the Random Forest
% algorithm and they are compared with the Ground Truth

%Random forest
predictions = MdlFinal.predict(input_test);

cellchar = string(cellstype);
predchar = string(predictions);

dim = size(predchar,1);
right = 0;
wrong = 0;
for i = 1:dim
    if(strcmp(predchar(i,1),cellchar(i,1)))
        right = right + 1;
    else
        wrong = wrong + 1;
    end
end

%% Accuracy
% in this section the accuracy of the algorithm is calculated in this
% method: the number of right prediction divided by the total number of the
% predicted samples.

accuracy_ds = (right/dim)*100;

