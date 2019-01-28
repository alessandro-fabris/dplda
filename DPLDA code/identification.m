function [ logliks , identities ] = identification(A, F, G,Sigma, mu, xGal, idsGal, xProbe, isIdentOS)
    %INPUTS
    %       A, F, G, Sigma, mu - model parameters
    %       xGal, xProbe - gallery and probe videos
    %       idsGal - ground truth for gallery videos
    %       isIdentOS - should we perform os or cs identification?
    %
    %OUTPUT
    %      logliks(iProb,iGal): loglikelihood of video iProb
    %            belonging to identity iGal, given gallery videos of identity iGal
    %            logliks(:,iGal+1) is an additional column if (isIdentOS)
    %            to store the marginal likelihood of probe vids alone, 
    %            which is useful to estimate likelihood of probe video depicting
    %            subject not in our database (out of gallery).
    %      identitites: deriving identification results


    fprintf('\n')
    disp('________IDENTIFICATION________')

    %% set up 'macro-videos' parameters
    
    sharedSpaceSize = size(F, 2);
    privateSpaceSize = size(G, 2);
    numSamplesProbe = size(xProbe,2);
    [numFeats, numClassesGal, numFrames] = Utils.getSizes(xGal);
    xProbe = xProbe - repmat(mu, 1, numSamplesProbe, numFrames);
    nonEmptyBucketsGal = Utils.findNonEmptyBuckets(xGal);

    [AStack, CStack, SigmaStack, GammaStack, ~] = Utils.stackMatrices(F, G, A, Sigma, mu, 1, numFrames, sharedSpaceSize, privateSpaceSize);
    %Set up time-varying model: A, the state matrix, changes when we have a
    %transition from video 1 of person i to video 2 of person i
    AStackTv = zeros(sharedSpaceSize + privateSpaceSize, sharedSpaceSize + privateSpaceSize, 2);
    AStackTv(:,:,1) = AStack;
    AStackTv(:,:,2) = [eye(sharedSpaceSize),                    zeros(sharedSpaceSize, privateSpaceSize);...
                      zeros(privateSpaceSize, sharedSpaceSize), zeros(privateSpaceSize, privateSpaceSize)];
    maxJ = max(nonEmptyBucketsGal);
    model = ones(1,maxJ*numFrames);
    for j = 1:maxJ-1
       model(j*numFrames+1) = 2; 
    end
    
    %% compute conditionalLogLiks

    [V, K] = Kalman.offlineFilter(AStackTv, CStack, GammaStack, SigmaStack, eye(privateSpaceSize + sharedSpaceSize), maxJ*numFrames, 0,'model',model);
    if (isIdentOS)
        logliks=zeros(numSamplesProbe, numClassesGal+1); %additional column for marginals
    else
        logliks=zeros(numSamplesProbe, numClassesGal);        
    end

    for b = nonEmptyBucketsGal   %'macro-videos' of the same length, share the same final variance
                                 % -if Aij=A-, and can be filtered together
                                 % in batches. Hence, this outer loop.
        disp (strcat('Bucket: ',num2str(b)))
        numClassesGal = size(xGal{b},2);
        XGalConc = zeros(numFeats,numClassesGal,b*numFrames); %XGal w/ vids of sam eperson concatenated
        for j = 1:b
            for t = 1:numFrames
                XGalConc(:,:,(j-1)*numFrames+t) = xGal{b}(:,:,j,t); 
                %Build 'macro-videos' joining vides of the same person head-to-tail
            end
        end
        XGalConc = XGalConc - repmat(mu,1,numClassesGal,b*numFrames);

        XFiltConcGal = Kalman.filterOfflineToOnlineBatch(XGalConc, zeros(privateSpaceSize+sharedSpaceSize, 1), K, AStackTv, CStack, 'model', model);
        initStateForProbe = XFiltConcGal(1:sharedSpaceSize,:,b*numFrames);
        disp('Filtered long gallery')

        d = diag(SigmaStack).^(-1);
        SigmaInv = diag(d);
        quadTerm = CStack'*SigmaInv*CStack;

        prevVar = [V(1:sharedSpaceSize,1:sharedSpaceSize,b*numFrames),     zeros(sharedSpaceSize,privateSpaceSize);...
                   zeros(privateSpaceSize,sharedSpaceSize),                eye(privateSpaceSize)];
        XFiltProbe = zeros(sharedSpaceSize+privateSpaceSize, numSamplesProbe, numClassesGal);    
        loglikTemp = zeros(numSamplesProbe, numClassesGal);
        for timeIndex = 1:numFrames
            disp (strcat('Timestep: ',num2str(timeIndex),'/',num2str(numFrames)))
            if (timeIndex == 1)
               initial = true; 
            else
                initial = false;
            end
            [Vt, Kt, SInv, logdetS] = Kalman.offlineUpdate(AStack, CStack, GammaStack, SigmaStack, prevVar, quadTerm, 1,'initial', initial, 'chol', 1);
            prevVar = Vt;
            currentObserv = xProbe(:,:,timeIndex);
            %for iGal=1:I
            parfor iGal = 1:numClassesGal 
                if (initial)
                    initState = [initStateForProbe(:,iGal); zeros(privateSpaceSize,1)];
                    prevState = repmat(initState,1,numSamplesProbe);
                else
                    prevState = AStack*XFiltProbe(:,:,iGal);
                end
                projErr = currentObserv - CStack*prevState;
                XFiltProbe(:,:,iGal) = prevState+Kt*projErr;
                quadTermLoglik = sum(projErr.*(SInv*projErr));
                loglikTemp(:,iGal) = loglikTemp(:,iGal) - 0.5*quadTermLoglik' - (numFeats/2)*log(2*pi) - logdetS/2;

            end
        end
        for iGal2 = 1:numClassesGal
            galTotIndex = idsGal{b}(iGal2);
            logliks(:,galTotIndex) = loglikTemp(:,iGal2);
        end
    end
    
    %% For OS ident: loglikelihood of probe videos alone
    if(isIdentOS)
        prevVar = eye(sharedSpaceSize+privateSpaceSize);
        for timeIndex = 1:numFrames
            disp (strcat('Timestep: ', num2str(timeIndex), '/', num2str(numFrames)))
            if (timeIndex==1)
                initial = true;
            else
                initial = false;
            end
            [Vt, Kt, SInv, logdetS] = offlineKalmanUpdate(AStack, CStack, GammaStack, SigmaStack, prevVar, quadTerm, 1, 'initial', initial, 'chol', 1);
            prevVar = Vt;
            
            if (initial)
                initState = zeros(sharedSpaceSize + privateSpaceSize, 1);
                prevState = repmat(initState, 1, numSamplesProbe);
            else
                prevState = AStack*XFiltProbe;
            end
            projErr = xProbe(:,:,timeIndex) - CStack*prevState;
            XFiltProbe = prevState + Kt*projErr;
            quadTermLoglik = sum(projErr.*(SInv*projErr));
            logliks(:, numClassesGal+1) = logliks(:, numClassesGal+1) - 0.5*quadTermLoglik' - (numFeats/2)*log(2*pi) - logdetS/2;
        end 
    end

    %%  Derive estimated identity for probeVideos

    identities=zeros(1,numSamplesProbe);
    for iProbe=1:numSamplesProbe
        [~ ,ind]=max(logliks(iProbe,:));
        identities(iProbe)=ind;
    end
end

