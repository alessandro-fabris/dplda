%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DPLDA closed set identification
%
% Selecting different values for sharedSpaceSizes (size of identity space),
% privateSpaceSizes (size of noise space), jProbe (index of video to be 
% used as probe) and maxEmIterations (max number of iterations of EM 
% algorithm), this script tries all possible combinations in performing 
% closed set identification, and stores the successRate for each of these 
% combinations in a conveniently sized array.

% A full run of the script will allow for intuitive choice of hyperparameters.
% Proper cross-validation is not implemented here

% The data, called xTrain and loaded from (dataName).mat, is in the format 
% xTrain=cell{1,maxJ}, where maxJ is the max number of videos per 
% class (person). Every person with j videos is saved inside X{j}. 
% In matlab-ese: xTtrain{j}=double(numFeatures,numPeople,j,numFrames)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

% dataName = 'dataFeatureExtraction'
dataName = 'dataRaw' %notice this dataset hasn't been frontalized, hence performance will be poor.
dataFolder = '../data/';

load([dataFolder, dataName, '.mat']); 
addpath('../PLDA code');

%% hyper parameters and derived quantities
bComputeLoglik = false; %check convergence of training (slows things down)
sharedSpaceSizes = [50]; %size of identity subspace
privateSpaceSizes = [16]; %size of accidental conditions subspace
maxEmIterations = [3]; %max EM iterations
probeVideoIdcs = [1]; %index of video to be used as probe
convThdEm = 1e-8; %convergence threshold for EM algorithm

numValsShared = length(sharedSpaceSizes);
numValsPrivate = length(privateSpaceSizes);
numValsMaxEm = length(maxEmIterations);
numValsProbeIdx = length(probeVideoIdcs);

successRates = zeros(numValsShared, numValsPrivate, numValsMaxEm, numValsProbeIdx);
bestRatePlot = -Inf;

for idxShared = 1:length(sharedSpaceSizes)
    for idxPrivate = 1:length(privateSpaceSizes)
        for idxMaxEm = 1:length(maxEmIterations)
            for idxProbe = probeVideoIdcs           
                %% split data into gallery (also used for training) and probe (only used for test)
                currSharedSize = sharedSpaceSizes(idxShared);
                currPrivateSize = privateSpaceSizes(idxPrivate);
                currMaxEmIters = maxEmIterations(idxMaxEm);
                progressString = 'currIdxProbe=%u, currSharedSize=%u, currPrivateSize=%u, currMaxEmIters=%u';
                sprintf(progressString, idxProbe, currSharedSize, currPrivateSize, currMaxEmIters)
%                 [currSharedSize, currPrivateSize, currIdxProbe, currMaxEmIters]
                
                [xGal, xProbe, idsGal, idsProbe] = Utils.getGalleryProbe(xTrain, idsTrain, idxProbe);
                [numFeats, ~, numFrames] = Utils.getSizes(xGal);
                                
                %% initialize at random
%                 seed = rng;
%                 %rng(seed)
%                 FInit = rand(numFeats, currPrivateSize);
%                 GInit = rand(numFeats, currSharedSize);
%                 SigmaInit = abs(diag(rand(1, numFeats)));
%                 % eig_ = 2*rand(1, currSharedSize)-1;
%                 eig_ = rand(1, currSharedSize);
%                 BM = orth(rand(currSharedSize));
%                 AInit = BM*diag(eig_)*BM';
                
                %% initialize with PLDA
                disp('Training PLDA for EM initialization...')
                [xPldaTrain, xPldaTrainIds] = Utils.preprocessDataPlda(xGal, idsGal, numFrames, false);
                pldaModel = PLDA_Train(xPldaTrain, xPldaTrainIds, currMaxEmIters, currSharedSize, currPrivateSize, 0, 0, 0);
                FInit = pldaModel.F;
                GInit = pldaModel.G;
                SigmaInit = diag(pldaModel.Sigma);
                % eig_=2*rand(1,Dw)-1;
                eig_ = rand(1,currPrivateSize);
                BM = orth(rand(currPrivateSize));
                AInit = BM*diag(eig_)*BM';
                disp('Training PLDA for EM initialization...Done')
                
                %% Training: em algorithm for parameter estimate
                muInit = Utils.getMean(xGal);
                [A, F, G, Sigma, mu, progressLoglik] = emEstimate(AInit, FInit, GInit, SigmaInit, muInit,...
                    xGal, currMaxEmIters, convThdEm, bComputeLoglik);
                if bComputeLoglik
                    figure; hold on; grid on; title('Loglikelihood vs Epoch')
                    plot(progressLoglik, 'LineWidth', 2);
                end
                
                %% Test: identification of probe videos
                % real identities (ground truth) are inside xProbeIds
                [conditionalLogLiksProbe, idsProbeEst] = identification(A, F, G, Sigma, mu, xGal,idsGal, xProbe, false);
                successfulIdentifications = find(idsProbe(:) == idsProbeEst(:));
                successRate = length(successfulIdentifications)/length(idsProbeEst);
                successRates(idxShared, idxPrivate, idxMaxEm, idxProbe) = successRate;
                if (successRate > bestRatePlot)
                    bestRatePlot = successRate;
                    idsProbeEstPlot = idsProbeEst;
                    idsProbeRealPlot = idsProbe;
                    idxProbePlot = idxProbe;
                end
            end
        end
    end
end

%Visualize features and result of identification
switch dataName
    case 'dataFeatureExtraction'
        dimx = 33;
        dimy = 62;
    case 'dataRaw'
        dimx = 34;
        dimy = 34;
    otherwise
        error 'Unknown dataSet'
end  
Utils.displayFaceMosaic(xTrain{1,4}(:,:,:,1), 'dimx', dimx, 'dimy', dimy, 'groundTruth', idsProbeRealPlot, 'estim', idsProbeEstPlot, 'idxProbe', idxProbePlot);
