function [A, F, G, Sigma, mu, overallLoglik] = emEstimate(AInit, FInit, GInit, SigmaInit, muInit,...
    xTrain, maxIter, thresh, computeLoglik )
    
    % Estimates DPLDA model parameters and if computeLoglik returns the
    % value of the loglikelihood at every iteration to verify that it keeps
    % increasing. Since it's expensive to do so, once sanity checks are
    % performed, computeLogLik may be set to 0.

    %It's optimized acording to the long 'macro-video' implementation and
    %thread-safe (parfor).

    %OUTPUT: estimated parameters and progress of loglikelihood
    %
    %INPUT: 
    %       AInit, FInit, GInit, SigmaInit, muInit - initial guess for param
    %       xTrain - training data
    %       I,J,T,f,Dh,Dw - hyperparameters
    %       maxIter,thresh - responsible for duration of learning phase
    %       coputeLogLik: if 1, we keep track of lerning progress by computing
    %               loglik at every step and making sure it keeps increasing. 
    %               Set to 0 for efficiency.
    %% Intitalization
    fprintf('\n')
    disp '________EM Algorithm________'
    
    sharedSpaceSize = size(FInit, 2);
    privateSpaceSize = size(GInit, 2);
    [numFeats, ~, framesPerVid] = Utils.getSizes(xTrain);
    overallLoglik = zeros(1,maxIter);
    mu = muInit;
    Sigma = (SigmaInit+SigmaInit')/2;
    A = AInit;
    G = GInit;
    F = FInit;
    nonEmptyBucketsTrain = Utils.findNonEmptyBuckets(xTrain);
    loglik = cell(1,length(xTrain));
    %initialize so parfor won't get mad
    Sinv = NaN;
    logDetS = NaN;
    for b = nonEmptyBucketsTrain
        I = size(xTrain{b}, 2);
        loglik{b} = zeros(I, 1);
    end
    maxJ = max(nonEmptyBucketsTrain);
    %keep track of time steps at which a transition takes place within a macrovideo
    isFrameNewVideo = zeros(1, maxJ*framesPerVid); 
    for j = 1:maxJ-1
       isFrameNewVideo(j*framesPerVid + 1) = 1; 
    end
    modelSelector = isFrameNewVideo + 1;    
    bConverged = false;
    numIters = 1;
    oldLoglik = -Inf;
    %% Cycle updating parameters at each iteration through ML-like criterion
    fprintf('\n')
    while (numIters <= maxIter && not(bConverged)) %either converged or reached maxNumber or iterations
        str = ['Iteration ',num2str(numIters),' out of (max) ',num2str(maxIter)];
        disp (str);

        [AStack, CStack, SigmaStack, GammaStack, ~] = Utils.stackMatrices(F, G, A, Sigma, mu, 1, framesPerVid, sharedSpaceSize, privateSpaceSize);
        %Set up time-varying model: A, the state matrix, changes when we have a transition from video 1 of person i to video 2 of person i
        AStackTv = zeros(sharedSpaceSize+privateSpaceSize, sharedSpaceSize+privateSpaceSize, 2);
        AStackTv(:,:,1) = AStack;
        AStackTv(:,:,2) = [eye(sharedSpaceSize), zeros(sharedSpaceSize, privateSpaceSize); zeros(privateSpaceSize, sharedSpaceSize), zeros(privateSpaceSize, privateSpaceSize)]; %transition matrix from a video to the following

        %offline Kalman filter to compute kalman gains and observation covariance matrix
        if not(computeLoglik)
            [V, K] = Kalman.offlineFilter(AStackTv, CStack, GammaStack, SigmaStack, eye(privateSpaceSize+sharedSpaceSize), maxJ*framesPerVid, 0, 'model', modelSelector);
        else
            [V, K, Sinv, logDetS] = Kalman.offlineFilter(AStackTv, CStack, GammaStack, SigmaStack, eye(privateSpaceSize+sharedSpaceSize), maxJ*framesPerVid, 1, 'model', modelSelector);
        end

        numB = cell(max(nonEmptyBucketsTrain), 1);
        denB = cell(max(nonEmptyBucketsTrain), 1);
        numA = cell(max(nonEmptyBucketsTrain), 1);
        denA = cell(max(nonEmptyBucketsTrain), 1);
        diagonal = cell(max(nonEmptyBucketsTrain), 1);
        
        %this is to compute Sigma, which has to be done after the parfor as it needs C
        VZSum=cell(max(nonEmptyBucketsTrain), 1); 
        for b = nonEmptyBucketsTrain
            numB{b} = zeros(numFeats, privateSpaceSize+sharedSpaceSize);
            denB{b} = zeros(privateSpaceSize+sharedSpaceSize);
            numA{b} = zeros(privateSpaceSize);
            denA{b} = zeros(privateSpaceSize);
            diagonal{b} = zeros(numFeats, 1);
            VZSum{b} = zeros(sharedSpaceSize+privateSpaceSize, sharedSpaceSize+privateSpaceSize);
        end

        for b = nonEmptyBucketsTrain
            % parfor b=nonEmptyBucketsTrain %it works!
            I = size(xTrain{b},2);
            XMacro = zeros(numFeats, I, b*framesPerVid);
            for j=1:b
                for t = 1:framesPerVid
                    XMacro(:, :, (j-1)*framesPerVid + t)= xTrain{b}(:, :, j, t);
                end
            end
            XMacro = XMacro - repmat(mu, 1, I, b*framesPerVid);
            [J, VZSmooth, VVZSmooth] = Kalman.offlineSmoother(AStackTv, GammaStack, V(:, :, 1:b*framesPerVid), 'model', modelSelector(1:b*framesPerVid));
            EZFilt = Kalman.filterOfflineToOnlineBatch(XMacro, zeros(privateSpaceSize+sharedSpaceSize, 1), K, AStackTv, CStack, 'model', modelSelector);
            EZSmooth{b} = Kalman.smootherOfflineToOnlineBatch(AStackTv, J, EZFilt, 'model', modelSelector(1:b*framesPerVid));
            if(computeLoglik)
                loglik{b} = Kalman.loglikOfflineToOnlineBatch(CStack, AStackTv, Sinv, logDetS, zeros(sharedSpaceSize+privateSpaceSize, 1), EZFilt, XMacro, 'model', modelSelector);
                if (any(loglik{b} == -Inf) || any(loglik{b} == Inf) || any(isnan(loglik{b})))
                    disp('Loglik is NaN or Inf, look into this')
                end
                
            end
            I = size(xTrain{b},2);
            for j = 1:b
                for t = 1:framesPerVid
                    numB{b} = numB{b} + XMacro(:, :, (j-1)*framesPerVid + t)*EZSmooth{b}(:, :, (j-1)*framesPerVid + t)';
                    denB{b} = denB{b} + EZSmooth{b}(:, :, (j-1)*framesPerVid + t)*EZSmooth{b}(:, :, (j-1)*framesPerVid + t)'...
                            + VZSmooth(:, :, (j-1)*framesPerVid + t)*I;
                    
                    if (t>1)
                        numA{b} = numA{b} + EZSmooth{b}(sharedSpaceSize + (1:privateSpaceSize), :, (j-1)*framesPerVid + t)*EZSmooth{b}(sharedSpaceSize + (1:privateSpaceSize), :, (j-1)*framesPerVid + t-1)'...
                                + VVZSmooth(sharedSpaceSize + (1:privateSpaceSize), sharedSpaceSize + (1:privateSpaceSize), (j-1)*framesPerVid + t)*I;
                        denA{b} = denA{b} + EZSmooth{b}(sharedSpaceSize + (1:privateSpaceSize), :, (j-1)*framesPerVid + t-1)*EZSmooth{b}(sharedSpaceSize + (1:privateSpaceSize), :, (j-1)*framesPerVid + t-1)'...
                                + VZSmooth(sharedSpaceSize + (1:privateSpaceSize), sharedSpaceSize + (1:privateSpaceSize), (j-1)*framesPerVid + t-1)*I;
                    end
                    VZSum{b} = VZSum{b} + VZSmooth(:, :, (j-1)*framesPerVid + t)*I ;
                end
            end
        end
        num = zeros(numFeats, privateSpaceSize + sharedSpaceSize);
        den = zeros(privateSpaceSize + sharedSpaceSize);
        for b = nonEmptyBucketsTrain
            num = num + numB{b};
            den = den + denB{b};
        end
        BEst = num/den;
        F = BEst(:, 1:sharedSpaceSize);
        G = BEst(:, sharedSpaceSize+1:sharedSpaceSize+privateSpaceSize);

        num = zeros(privateSpaceSize);
        den = zeros(privateSpaceSize);
        for b = nonEmptyBucketsTrain
            num = num + numA{b};
            den = den + denA{b};
        end
        A = num/den;

        %Once B is computed, I can compute Sigma, not before. Would probably converge anyway?
        diagonal = zeros(numFeats, 1);
        for b = nonEmptyBucketsTrain
            I = size(xTrain{b}, 2);
            EZZSum = zeros(sharedSpaceSize + privateSpaceSize, sharedSpaceSize + privateSpaceSize);
            for j = 1:b
                for t = 1:framesPerVid
                    diagonal = diagonal + dot((reshape(xTrain{b}(:, :, j, t), numFeats, I) - repmat(mu, 1, I)), (reshape(xTrain{b}(:, :, j, t), numFeats, I) - repmat(mu, 1, I)), 2)...
                             - dot(2*BEst*EZSmooth{b}(:, :, (j-1)*framesPerVid + t), (reshape(xTrain{b}(:, :, j, t), numFeats, I) - repmat(mu,1,I)),2);
                    EZZSum = EZZSum+EZSmooth{b}(:, :, (j-1)*framesPerVid + t)*EZSmooth{b}(:, :, (j-1)*framesPerVid + t)';
                end
            end
            diagonal = diagonal + dot(BEst*(EZZSum + VZSum{b} ), BEst, 2);
        end
        numVids = Utils.totNumVids(xTrain);
        Sigma = diag(diagonal)./(numVids*framesPerVid);
        
        if (computeLoglik)
            overallLoglik(numIters)=0;
            for b = nonEmptyBucketsTrain
                overallLoglik(numIters) = sum(loglik{b});
            end
            bConverged = Utils.emConverged(overallLoglik(numIters), oldLoglik, thresh);
            if bConverged
                overallLoglik = overallLoglik(1:numIters);
            end
            oldLoglik = overallLoglik(numIters);
        end
        numIters = numIters+1;
    end
end

