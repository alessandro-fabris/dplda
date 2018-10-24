% Example of face verification test using a learned PLDA model
%   - Verify whether two images are from the same identity or not.
%
% In this example we evaluate two images from NData1 identities and compare
% each image pairs. So there are NData1 positive pairs (match) and 
% NData1 * (NData1 - 1) / 2  negative pairs (not-match). The total number
% of verification test is then NData1 * (NData1 + 1) / 2.
% 
%
%******************** Disclaimer *****************************************
%*** This program is distributed in the hope that it will be useful, but
%*** WITHOUT ANY WARRANTY; without even the implied warranty of 
%*** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%*** Feel free to use this code for academic purposes.  Plase use the
%*** citation provided below.
%
% S.J.D. Prince and J.H. Elder, “Probabilistic linear discriminant analysis
% for inferences about identity,”  ICCV, 2007. 
%
% P. Li and S.J.D. Prince, “Probabilistic Methods for Face Registration and
% Recognition”, In Advances in Face Image Analysis: Techniques and
% Technologies, Y. Zhang (eds.)  (in press). 
%
%**************************************************************************
%
% 04-03-2010
% This demo takes about 16 seconds with current setting on a computer with
% Indel Xeon CPU @ 2.5GHz and 2.5GB of RAM.
tic

NData1 = 10; % Number of test images (2~100 in the example here)

% Load test data (100 gallery images and 100 probe images)
load('Data.mat', 'DataGallery', 'DataProbe');

% Load the learned PLDA model
load('PLDAModel.mat', 'PLDAModel');
   
fprintf('Testing...\n');

% Result on positive and negative classes
LoglikeRatioPos = zeros(NData1, 1);             % Positive class
LoglikeRatioNeg = zeros((NData1 - 1) * NData1 / 2, 1); % Negative class
t = 0;
% Verification test by model comparison
for i = 1 : NData1
    LoglikeRatioPos(i) = PLDA_Verification(PLDAModel, ...
            DataGallery(:, i), DataProbe(:, i));    
    for j = i + 1 : NData1
        t = t + 1;        
        LoglikeRatioNeg(t) = PLDA_Verification(PLDAModel, ...
            DataGallery(:, i), DataProbe(:, j));    
    end
end

% ROC curve
RMax = max([LoglikeRatioPos', LoglikeRatioNeg']);
RMin = min([LoglikeRatioPos', LoglikeRatioNeg']);
IndexThreshold = RMin:RMax;

ROC = zeros(length(IndexThreshold), 2);
for i = 1 : length(IndexThreshold)
    TruePositive = length(find(LoglikeRatioPos > IndexThreshold(i)))...
        ./ length(LoglikeRatioPos);
    FalsePositive = length(find(LoglikeRatioNeg > IndexThreshold(i)))...
        ./ length(LoglikeRatioNeg);
    
    ROC(i, 1) = FalsePositive;
    ROC(i, 2) = TruePositive;
end

figure(1);
h = plot(ROC(:, 1), ROC(:, 2), 'r.-');
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC curve of face verification');
toc

