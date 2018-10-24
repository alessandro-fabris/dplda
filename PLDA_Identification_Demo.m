% Example of face identification using a learned PLDA model
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
% This demo takes about 2.5 seconds with current setting on a computer with
% Indel Xeon CPU @ 2.5GHz and 2.5GB of RAM.

tic
% Load test data (100 gallery images and 100 probe images)
load('Data.mat', 'DataGallery', 'DataProbe');

% Load the learned PLDA model
load('PLDAModel.mat', 'PLDAModel');
   
fprintf('Testing...\n');

% Identification test by model comparison
LoglikeTest = PLDA_Identification(PLDAModel, DataGallery, DataProbe);    

% Evalautet the recognition rate
[maxPost Label] = max(LoglikeTest);
IndexCorrect = find(1:size(LoglikeTest, 1)==Label);
totalCorrect = length(IndexCorrect);
% Store the result
CR = totalCorrect / size(LoglikeTest, 1);

% The identification result
fprintf('First rank identification rate based on %d factors is %1.2f\n', ...
    size(PLDAModel.F, 2), CR);

% Show the log-likelihood matrix of matching the probe to gallery images
% The diagonal elements should have higher value than the others.
figure(1);
imagesc(LoglikeTest);
colorbar
xlabel('Probe images');
ylabel('Gallery images');
title('Log likelihood matrix of matching probe images to gallery images');
toc

