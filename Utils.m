classdef Utils
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % generic utility functions shared throughout the code
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    methods(Static)
        
        function [h, tableImages ] = displayFaceMosaic(X, varargin)
            numRowsMosaic = size(X,3);
            numColsMosaic = size(X,2);
            dimxImg = round(sqrt(size(X,1)));
            dimyImg = dimxImg;
            
            args = varargin;
            nargs = length(args);
            for j = 1:2:nargs
                switch args{j}
                    case 'dimx',            dimxImg = args{j+1};
                    case 'dimy',            dimyImg = args{j+1};
                    case 'groundTruth',     groundTruth = args{j+1};
                    case 'estim',           estim = args{j+1};
                    case 'idxProbe',        idxProbe = args{j+1};
                    otherwise, error(['unrecognized argument ' args{j}])
                end
            end
            bPlotIdent = exist('groundTruth', 'var') || exist('estim', 'var') || exist('idxProbe', 'var');
            if (bPlotIdent)
               assert(exist('groundTruth', 'var') && exist('estim', 'var') && exist('idxProbe', 'var'), 'Missing something for identification plot') 
            end
            
            % Allow for padding
            pad = 3;
            tableImages = zeros(pad + numRowsMosaic * (dimyImg + pad), ...
                pad + numColsMosaic * (dimxImg + pad));
            % Images
            for i = 1:numRowsMosaic
                for j = 1:numColsMosaic
                    tableImages(pad + (i - 1) * (dimyImg + pad) + (1:dimyImg), ...
                        pad + (j - 1) * (dimxImg + pad) + (1:dimxImg)) = ...
                        reshape(X(:,j,i), dimyImg, dimxImg);
                end
            end
            % Allow for color
            isNormalized = max(tableImages(:)) <= 1;
            colorTableImage = repmat(tableImages, [1 1 3]);
            if not(isNormalized)
                colorTableImage = colorTableImage/255;
            end
            % If required, visualize correct/wrong identification
            if (bPlotIdent)
                frameWidth = 1;
                for idxCol = 1:numColsMosaic
                   if (groundTruth(idxCol) == estim(idxCol)) 
                       color = [0,1,0];
                   else
                       color = [1,0,0];
                   end
                   colorTableImage = Utils.frameImage(colorTableImage, idxProbe, idxCol, frameWidth, dimxImg, dimyImg, pad, color);
                end
            end
            figure;
            h = image(colorTableImage);
            axis image off
            
            % Stress gal vs probe
            if (bPlotIdent)
               yl = ylim;
               xl = xlim;
               textDist = (yl(2) - yl(1))/numRowsMosaic;
               for idxRow = 1:numRowsMosaic
                   x = xl(2) + (xl(2) - xl(1))/100;
                   y = textDist/2 + (idxRow-1)*textDist;
                   if idxRow == idxProbe
                       txt = 'Probe';
                   else
                       txt = 'Gallery';
                   end
                   text(x, y, txt);
               end
            end
        end
        
        function moisaicOut = frameImage(mosaicIn, mosaicRow, mosaicCol, frameWidth, dimxImg, dimyImg, padWidth, color)
            dimensions = size(mosaicIn);
            assert(dimensions(3)==3);
            moisaicOut = mosaicIn;
            frameLengthVert = frameWidth*2 + dimyImg;
            frameLengthHor = frameWidth*2 + dimxImg;
            vertSegment = permute(repmat(color(:), [1, frameLengthVert, frameWidth]), [2,3,1]);
            horSegment = permute(repmat(color(:), [1, frameWidth, frameLengthHor]), [2,3,1]);
            %left
            rowIdcs = padWidth - frameWidth + (mosaicRow - 1) * (dimyImg + padWidth) + (1:frameLengthVert);
            colIdcs = padWidth - frameWidth + (mosaicCol - 1) * (dimxImg + padWidth) + (1:frameWidth);
            moisaicOut(rowIdcs, colIdcs, :) = vertSegment;
            %right
            colIdcs = colIdcs + dimxImg + frameWidth;
            moisaicOut(rowIdcs, colIdcs, :) = vertSegment;
            %top
            rowIdcs = padWidth - frameWidth + (mosaicRow - 1) * (dimyImg + padWidth) + (1:frameWidth);
            colIdcs = padWidth - frameWidth + (mosaicCol - 1) * (dimxImg + padWidth) + (1:frameLengthHor);
            moisaicOut(rowIdcs, colIdcs, :) = horSegment;
            %bottom
            rowIdcs = rowIdcs + dimyImg + frameWidth;
            moisaicOut(rowIdcs, colIdcs, :) = horSegment;
        end
                
        function [numFeats, numClasses, numFrames] = getSizes(xAll)
            % INPUT:
            %   xAll{J} contains all the people with exactly J videos. X_all{J}(:,i,j,t) is the
            %               t-th frame of the j-th video of the i-th person (among the
            %               ppl depicted in exactly J videos).
            % OUTPUT
            %   numFeats: number of features.
            %   numClasses: number of different classes (individuals)
            %   numFrames: number of frames per video
            
            numClasses = 0;
            numFeats = NaN;
            numFrames = NaN;
            for idxBucket = 1:size(xAll,2)
                x = xAll{idxBucket};
                if not(size(x) == 0)
                    numClasses = numClasses + size(x,2);
                    if (isnan(numFeats))
                        assert(isnan(numFrames))
                        numFeats = size(x,1);
                        numFrames = size(x,4);
                    else
                        %we assume numFeat and numFrames are consistent across buckets
                        assert(numFeats == size(x,1));
                        assert(numFrames == size(x,4));
                    end
                end
            end
            
        end
        
        % this function is highly inefficient but in the grand scheme of things 
        % it doesn't matter and makes me feel safe with all these dimensions at play
        function mu = getMean(x)
            count = 0;
            [numFeats, ~, numFrames] = Utils.getSizes(x);
            nonEmptyBuckets = Utils.findNonEmptyBuckets(x);
            
            sum = zeros(numFeats,1);
            for b = nonEmptyBuckets
                temp = x{b};
                for i = 1:size(temp,2)
                    for j = 1:b
                        for t=1:numFrames
                            sum = sum + temp(:, i, j, t);
                            count = count+1;
                        end
                    end
                end
            end
            mu = sum/count;
        end
        
        function n = totNumVids(x)
            n=0;
            nonEmptyBuckets = Utils.findNonEmptyBuckets(x);
            for b = nonEmptyBuckets
                numClasses = size(x{b},2);
                n = n + numClasses*b;
            end
        end
        
        function n = findNonEmptyBuckets(x)
            numBuckets = size(x,2);
            n = [];
            for idxBucket = 1:numBuckets
                if not(isempty(x{idxBucket}))
                    n = [n,idxBucket];
                end
            end
        end
        
        function [AStack, CStack, SigmaStack, GammaStack, muStack] = stackMatrices(F, G, A, Sigma, mu, AugmentFactor, T, Dh, Dw)
            % OUTPUT: 4 'global' matrices describing the dynamics of the
            % augmented, (Dh+State_Augment*Dw)-size, system
            % INPUT:  F h-to-x matrix
            %         G w-to-x matrix
            %         A state evolution matrix
            %         Sigma observation noise matrix
            %         J: number of videos per person
            %         AugmentFactor: how many vectors are we stacking up?
            %               normally State_augment=J, but there can be
            %               exceptions
            %         Aij = 1 if different state evlution matrix per video
            
            GDiag=G;
            for i = 1:AugmentFactor-1
                GDiag = blkdiag(GDiag, G);
            end
            CStack = [repmat(F, AugmentFactor, 1), GDiag]; %size: State_Augment*fx(JDw+Dh)
            
            AStack = eye(Dh);
            for i = 1:AugmentFactor
                AStack = blkdiag(AStack, A); %final size: (Dh+State_Augment*Dw)x(Dh+State_Augment*Dw)
            end
            
            SigmaStack = Sigma;
            for i = 1:AugmentFactor-1
                SigmaStack = blkdiag(SigmaStack, Sigma);%final size: State_Augment*fxState_Augment*f
            end
            
            GammaStack = zeros(Dh);
            for i = 1:AugmentFactor
                GammaStack = blkdiag(GammaStack,eye(Dw));% final size: (Dh+State_Augment*Dw)x(Dh+State_Augment*Dw)
            end
            
            muStack=repmat(mu, AugmentFactor, T);
            
        end
        
        function [converged, decrease] = emConverged(loglik, previousLoglik, threshold, checkIncreased)
            % Implemented by Kevin Murphy
            % EM_CONVERGED Has EM converged?
            % [converged, decrease] = em_converged(loglik, previous_loglik, threshold)
            %
            % We have converged if the slope of the log-likelihood function falls below 'threshold',
            % i.e., |f(t) - f(t-1)| / avg < threshold,
            % where avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log lik at iteration t.
            % 'threshold' defaults to 1e-4.
            %
            % This stopping criterion is from Numerical Recipes in C p423
            %
            % If we are doing MAP estimation (using priors), the likelihood can decrase,
            % even though the mode of the posterior is increasing.
            
            if nargin < 3, threshold = 1e-4; end
            if nargin < 4, checkIncreased = 1; end
            
            converged = false;
            decrease = 0;
            
            if checkIncreased
                if loglik - previousLoglik < -1e-3 % allow for a little imprecision
                    fprintf(1, '******likelihood decreased from %6.4f to %6.4f!\n', previousLoglik, loglik);
                    decrease = 1;
                    converged = false;
                    return;
                end
            end
            
            deltaLoglik = abs(loglik - previousLoglik);
            avgLoglik = (abs(loglik) + abs(previousLoglik) + eps)/2;
            if (deltaLoglik / avgLoglik) < threshold
                converged = true;
            end
        end
        
        
        function [xGal, xProbe, xGalIds, xProbeIds] = getGalleryProbe(xAll, xAllIds, jProbe, varargin)
        % this function devides data into gallery (possibly also used for training) 
        % and probe
        % INPUT:
        %   {J} contains all the people with exactly J videos. xAll{J}(:,i,j,t) is the
        %               t-th frame of the j-th video of the i-th person (among the
        %               ppl depicted in exactly J videos)
        %   xAllIds{J} contains the numeric IDs of the ppl with exctly J videos.
        %               xAllIds{j}(i) contains the ID of the i-th person
        %   jProbe determines which video will be used as a probe for all the subjects.

        % OUTPUT
        %   xGal: gallery data, with structure analogous to X_all
        %   xGalIds : IDs with the same structure as above
        %   xProbe: probe videos of size numFeatures x numPeople x T
        %   xProbeIds : ID of the probe videos
        
            args = varargin; %parse optional arguments
            nargs = length(args);
            numExtVids = 0;
            xExt = NaN;
            
            for i=1:2:nargs
              switch args{i}
                  case 'xExt', xExt = args{i+1};
                  case 'numExt', numExtVids=args{i+1};
               otherwise, error(['unrecognized argument ' args{i}])
              end
            end

            [numFeats, numClassesInt, numFrames] = Utils.getSizes(xAll);
            numVidsProbe = numClassesInt + numExtVids;
            xProbe = zeros(numFeats, numVidsProbe, numFrames);
            xProbeIds = NaN(numVidsProbe,1);
            
            numBucketsAll = size(xAll,2);
            numBucketsGallery = numBucketsAll - 1;
            xGal = cell(1 ,numBucketsGallery);
            xGalIds = cell(1, numBucketsGallery);
            countGalVids = 0;
            
            for idxBucket = 1:numBucketsAll
                allVidIdcs = 1:idxBucket;
                vidIdcsTrain = allVidIdcs(not(allVidIdcs==jProbe));
                vids = xAll{idxBucket};
                numClassesInBucket = size(vids,2);
                for idxClass = 1:numClassesInBucket
                    countGalVids = countGalVids + 1;
                    xProbe(:,countGalVids,:) = vids(:,idxClass, jProbe, :);
                    xProbeIds(countGalVids) = xAllIds{idxBucket}(idxClass);
                end
                if (not(isempty(vids)) && idxBucket > 1)
                    xGal{idxBucket-1} = vids(:,:,vidIdcsTrain,:);
                    xGalIds{idxBucket-1} = xAllIds{idxBucket};
                end
            end
            
            assert(countGalVids == numClassesInt, 'Couldn t select 1 probe video for each class')
            
            if (numExtVids > 0)
                assert( not( isnan(xExt)), 'Need a structure holding external videos in order to add them')
                indStartExt = (jProbe - 1)*numExtVids + 1; %as jProbe varies, select a different set of external videos
                xProbe(:, countGalVids:end, :) = xExt(:, indStartExt + (1:numExtVids),:);
                xProbeIds(countGalVids:end) = numClassesInt + 1;               
            end
        end
        
                
        
        function [ frames , frameIds ] = preprocessDataPlda(x, xIds, framesPerVid, bRandSel)
        % Given video data in a format convenient for DPLDA converts into a
        % image-set representation, convenient for PLDA functions.

        %       INPUT
        %       x: gallery images
        %       xIds: identities associated to images in consistent format
        %       framesPerVid: number of images we want to use per video
        %       bRandSel: if (b_rand) we pick N videos at random, if !(rand) we pick the
        %       first N.
        %
        %       OUTPUT
        %       frames: nFeature x nSample  -  Training data
        %       frameIds: nSample x nIdentity  -  Identity matrix of training data
        
        [numFeats, numIdents, numFrames] = Utils.getSizes(x);
        numVid = Utils.totNumVids(x);


        frames = NaN(numFeats, numVid*framesPerVid);
        rows = []; %sample
        cols = []; %identity

        nonEmptyBuckets = Utils.findNonEmptyBuckets(x);
        count = 0;
        for b = nonEmptyBuckets
            numClassesInBucket = size(x{b},2);
            for idxClass = 1:numClassesInBucket   
               for idxVid = 1:b
                   idcsSelection = (1:framesPerVid);
                   if bRandSel
                       perm = randperm(numFrames);
                       idcsSelection = perm(1:framesPerVid);
                   end
                  for idxFrame = idcsSelection
                      count = count+1;
                      frames(:, count)= x{b}(:, idxClass, idxVid, idxFrame);
                      rows = [rows, count];
                      ident = xIds{b}(idxClass);
                      cols = [cols,ident];          
                  end
               end    
            end
        end
        frameIds = sparse(rows, cols, ones(1,numVid*framesPerVid), numVid*framesPerVid, numIdents);
        end
        
    end
end