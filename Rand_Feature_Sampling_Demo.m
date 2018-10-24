% Example of face identification using an ensemble of PLDA models trained
% by random subset of raw pixel intensities of the face image.
%
%******************** Disclaimer *****************************************
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%
% The program is free for academic use. If you are interested in using the
% software for commercial purposes.  Please contact
%   Dr. Simon J. D. Prince
%   Email: s.prince@cs.ucl.ac.uk
%
% Plase use the citation provided below if it is useful to your research:
%
% S.J.D. Prince and J.H. Elder, “Probabilistic linear discriminant analysis
% for inferences about identity,”  ICCV, 2007.
%
% P. Li and S.J.D. Prince, “Probabilistic Methods for Face Registration and
% Recognition”, In Advances in Face Image Analysis: Techniques and
% Technologies, Y. Zhang (eds.)  (in press).
%
%**************************************************************************
% Date: 04-03-2010
%
% This demo takes about 37 seconds (10 subsets) on a computer with Indel
% Xeon CPU @ 2.5GHz and 2.5GB of RAM.

function Rand_Feature_Sampling_Demo
tic

% Number of iterations to train a PLDA model
N_ITER = 6;

% Subspace dimension to evaluate (1 ~ 195) in this example
N_FAC_EST = 64;

% Dimension of randomly selected feature subset
N_DIM = 100;

% Number of random feature subsets in the ensemble. The larger the number,
% the more time to train, but the better the performance and the stability.
NTrial = 10;

% Get training and test data
load('Data.mat', 'DataTrain', 'ImageID', 'DataGallery', 'DataProbe');

N_Test = size(DataGallery, 2);

% Training
LoglikeTest = zeros(N_Test, N_Test);
CRS = zeros(NTrial, 1); % Recognition rate of each individual PLDA
CR = zeros(NTrial, 1);  % Recognition rate of the PLDA ensemble
for i = 1 : NTrial
    fprintf('Estimating %d factors of grid %d, training ...\n',...
        N_FAC_EST, i);
    
    % Randomly select a subset of features from the full image for
    % training and test
    RandDim = randperm(size(DataTrain, 1));
    IndexDim = RandDim(1:N_DIM);
    
    DataTrainThis = DataTrain(IndexDim, :);
    DataGalleryThis = DataGallery(IndexDim, :);
    DataProbeThis = DataProbe(IndexDim, :);
    
    % Train PLDA
    PLDAModel = PLDA_Train(DataTrainThis, ImageID, ...
        N_ITER, N_FAC_EST, N_FAC_EST);
    
    % Testing
    LoglikeTestThis = PLDA_Identification(PLDAModel,...
        DataGalleryThis, DataProbeThis);
    
    % Evalautet the recognition rate
    CRS(i) = Correct_Recognition_Rate(LoglikeTestThis);
    
    % Accummulate the loglikelihood of each grid
    LoglikeTest = LoglikeTest + LoglikeTestThis;
    
    CR(i) = Correct_Recognition_Rate(LoglikeTest);
    fprintf('Grid %d of 100, recognition rate of this PLDA is %1.2f and that of ensemble is %1.2f\n',...
        i , CRS(i), CR(i));
    toc
end

fprintf('First rank identification rate of the PLDA ensemble based on %d factors is %1.2f\n', ...
    N_FAC_EST, CR(end));

% Show the log-likelihood matrix of matching the probe to gallery images
% The diagonal elements (true match) should have higher value than the
% others (true not-match). 
figure(1);
imagesc(LoglikeTest);
colorbar
xlabel('Probe images');
ylabel('Gallery images');
title('Log likelihood matrix of matching probe images to gallery images');


% Show the log-likelihood matrix of matching the probe to gallery images
% The diagonal elements (true match) should have higher value than the
% others (true not-match). 
figure(2);
h(1) = plot(CR, 'ro-');
hold on
h(2) = plot(CRS, 'ks--');
xlabel('Order of PLDA in the ensemble');
ylabel('First rank identification rate');
legend(h, 'PLDA ensemble', 'Individual PLDA');
title('Performance of PLDA ensemble and each individual PLDA');

toc

% Calculate first rank identification rate given the loglikelihood matrix
function CR = Correct_Recognition_Rate(LoglikeTest)
[maxPost Label] = max(LoglikeTest);
IndexCorrect = find(1:size(LoglikeTest, 1)==Label);
totalCorrect = length(IndexCorrect);
% Store the result
CR = totalCorrect / size(LoglikeTest, 1);