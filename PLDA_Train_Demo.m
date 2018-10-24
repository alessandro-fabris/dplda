% Example to train a Probabilistic Linear Discriminant Analysis(PLDA) model
%
% Note:: Runing this demo will train a new PLDA model and overwrite the
% current model file: PLDAModel.mat.
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
% This demo takes about 220 seconds with current setting on a computer with
% Indel Xeon CPU @ 2.5GHz and 2.5GB of RAM.
% 

tic

% Number of iterations to train a PLDA model
N_ITER = 6;

% Subspace dimensions to evaluate (1 ~ 195) in this example
N_FAC_EST = 64;

% Get training and test data
load('Data.mat', 'DataTrain', 'ImageID', 'DataGallery', 'DataProbe');

[N_DATA N_PERSON] = size(ImageID);

fprintf('Estimating PLDA model with %d factors\n', N_FAC_EST);

% Train the PLDA model
PLDAModel = PLDA_Train(DataTrain, ImageID, N_ITER, N_FAC_EST, N_FAC_EST);
   
% Store the model
save('PLDAModel.mat', 'PLDAModel');
toc

