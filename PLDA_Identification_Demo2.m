% Example of face identification using learned PLDA models with different
% subspace dimensions. This produces the curve shown in the book chapter.
%
%   P. Li and S.J.D. Prince, “Probabilistic Methods for Face Registration
%   and Recognition”, In Advances in Face Image Analysis: Techniques and
%   Technologies, Y. Zhang (eds.)  (in press).
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
% This demo takes about 15 seconds with current setting on a computer with
% Indel Xeon CPU @ 2.5GHz and 2.5GB of RAM.

tic

% Load test data (100 gallery images and 100 probe images)
load('Data.mat', 'DataGallery', 'DataProbe');

% Load the learned PLDA model
load('PLDAModels.mat', 'PLDAModels');

fprintf('Testing...\n');
N_Dim = zeros(length(PLDAModels), 1);
CR = zeros(length(PLDAModels), 1);
for i = 1 : length(PLDAModels);
    PLDAModel = PLDAModels{i};          % PLDA model
    N_Dim(i) = size(PLDAModel.F, 2);    % Subspace dimension
    
    
    % Identification test by model comparison
    LoglikeTest = PLDA_Identification(PLDAModel, DataGallery, DataProbe);
    
    % Evalautet the recognition rate
    [maxPost Label] = max(LoglikeTest);
    IndexCorrect = find(1:size(LoglikeTest, 1)==Label);
    totalCorrect = length(IndexCorrect);
    % Store the result
    CR(i) = totalCorrect / size(LoglikeTest, 1);
    
    % The identification result
    fprintf('First rank identification rate based on %d factors is %1.2f\n', ...
        N_Dim(i), CR(i));
end

% Show identification result on different subspace dimension.
figure(1);
h1(1) = plot(N_Dim, CR, 'rd-','Linewidth',2.5);

xlabel('Subspace dimension', 'FontSize', 12);
ylabel('Identification rate', 'FontSize', 12);
legend(h1, 'PLDA', 0);

set(gcf,'color',[1 1 1]);
set(gca,'Box','Off');

axis([0 130 0.4 0.95]);
hold off
toc

