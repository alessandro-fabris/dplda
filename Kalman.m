classdef Kalman
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % functions related to Kalman inference
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    methods(Static)
        
        function [V, K, varargout] = offlineFilter(A, C, Q, R, VInit, T, computeLogDet, varargin)
            % INPUTS:
            % A - the system matrix
            % C - the observation matrix
            % Q - the system covariance
            % R - the observation covariance
            % VInit - the initial state covariance
            % T - length of filtering horizon
            % computeLogDet - binary val that determines whether we need to compute the
            %        inverse of the innovation covariance matrix and the logarithm of
            %        its determinant.
            %
            % OPTIONAL INPUTS (string/value pairs [default in brackets])
            % 'model' - allows for a time-varying model wherein A=A(t).
            %     model(t)=m means use params from model m at time t [ones(1,T)]
            %     In this case, matrix A takes an additional final dimension,
            %     i.e., A(:,:,m).
            % 'usePrev' - binary val. If equal to 1 exploits VInit in a different
            %       way, basically considering it as an actual estimate of the previous
            %       variance, not as a prior on initial process variance.
            %
            % OUTPUTS (where X is the hidden state being estimated)
            % V(:,:,t) = Cov[X(:,t) | y(:,1:t)]
            % K(:,:,t) = Kalman gain at time t
            % Sinv(:,:,t) = inv(Covar( y(:,t)|y(:,1:t-1) ))
            % logDetS(:,t) = log[ det(Covar( y(:,t)|y(:,1:t-1) )) ]

            os  = size(C,1); %size of observations
            ss = size(A,1); % size of state space
            
            % set default params
            model = ones(1,T);
            usePrevResults = 0;
            
            args = varargin; %parse optional arguments
            nargs = length(args);
            for i = 1:2:nargs
                switch args{i}
                    case 'model',   model = args{i+1};
                    case 'usePrev', usePrevResults=args{i+1};
                    otherwise, error(['unrecognized argument ' args{i}])
                end
            end
            
            if (computeLogDet)
                Sinv = zeros(os, os, T);
                logDetS = zeros(1, T);
            end
            K = zeros(ss, os, T);
            V = zeros(ss, ss, T);
            for t = 1:T
                m = model(t);
                if t == 1
                    prevV = VInit;
                    bInitial = true;
                    if (usePrevResults)
                        bInitial = false;
                    end
                else
                    prevV = V(:,:,t-1);
                    bInitial = false;
                end
                d = diag(R).^(-1);
                RInv = diag(d);
                quadTerm = C'*RInv*C;
                if (computeLogDet)
                    [V(:,:,t), K(:,:,t), Sinv(:,:,t), logDetS(:,t)] = ...
                        Kalman.offlineUpdate(A(:,:,m), C, Q, R, prevV, quadTerm, computeLogDet, 'initial', bInitial, 'chol',true);
                else
                    [V(:,:,t), K(:,:,t),~,~] = ...
                        Kalman.offlineUpdate(A(:,:,m), C, Q, R, prevV, quadTerm, computeLogDet, 'initial', bInitial, 'chol', true);
                end                
            end
            if (computeLogDet)
                varargout{1} = Sinv;
                varargout{2} = logDetS;
            end
        end
        
        
        function [VNew, K, SInv, logDetS] = offlineUpdate(A, C, Q, R, V, quadTerm, computeLogDet, varargin)
            % KALMAN_UPDATE Do a one step update of the Kalman filter
            %
            % INPUTS:
            % A - the system matrix
            % C - the observation matrix
            % Q - the system covariance
            % R - the observation covariance
            % V(:,:) - Cov[X | y(:, 1:t-1)] prior covariance
            % quadTerm - quadratic term computed offline for efficency
            % computeLogDet: boolen that specifies whether we need to compute logDetS
            %           or not.
            %
            % OPTIONAL INPUTS (string/value pairs [default in brackets])
            % 'initial' - 1 means x and V are taken as initial conditions (so A and Q are ignored) [0]
            % 'chol' - boolean that specifies whether or not to use a cholesky
            % decomposition in computing log_detS. If stata space has high dimensionality 
            % it will most likely yield better stability. [false]
            %
            % OUTPUTS (where X is the hidden state being estimated)
            %  Vnew(:,:) = Var[ X(t) | y(:, 1:t) ]
            %  K(:,:) = kalman gain at time t
            %  Sinv = Sinv(:,:,t) = inv(Covar( y(:,t)|y(:,1:t-1) ))
            %  detS(:,t) = det(Covar( y(:,t)|y(:,1:t-1) ))
            
            
            % set default params
            bInitial = false;
            useChol = false;
            %parse optional params
            args = varargin; 
            for i = 1:2:length(args)
                switch args{i}
                    case 'initial', bInitial = args{i+1};
                    case 'chol', useChol = args{i+1};
                    otherwise, error(['unrecognized argument ' args{i}])
                end
            end
          
            if bInitial
                Vpred = V;
            else
                Vpred = A*V*A' + Q;
                Vpred=(Vpred + Vpred')/2; %force symmetry (which might be lost due to roundings)
            end
            
            %faster computation exploiting diagonal covar matrix
            d = diag(R).^(-1);
            RInv = diag(d);
            %invV = inv(Vpred);
            invV = pinv(Vpred);
            SInv = RInv - (d*d').*(C/(invV + quadTerm)*C');
            SInv = (SInv + SInv')/2;
            ss = length(V);
            logDetS = NaN;
            if(computeLogDet)
                if (not(useChol))
                    S = C*Vpred*C' + R;
                    S = (S+S')/2;
                    determinant = abs(det(S));
                    logDetS = log(determinant);
                    
                    if (determinant == Inf || determinant == 0)
                        detReg = determinant;
                        regTerm = 1;
                        if (determinant == Inf)
                            regOp = @(x)(x/2);
                        else
                            regOp = @(x)(x*2);
                        end
                        while (detReg == determinant)
                            regTerm = regOp(regTerm);
                            detReg = abs(det(S*regTerm));
                        end
                        logDetS = log(detReg) - length(S)*log(regTerm);
                    end
                    %the method below is quite slower
                    %     E=eig(C);
                    %     logDet2=sum(log(E))
                else
                    logDetS = -Kalman.logDet(SInv,'chol');
                end
                
            end
            
            if(abs(logDetS)==Inf || abs(logDetS)==-Inf)
                warning('Something wrong with eigenvalues?')
                keyboard
            end
            K = Vpred*C'*SInv; % Kalman gain matrix           
            VNew = (eye(ss) - K*C)*Vpred;
            VNew = (VNew + VNew')/2;
        end
        
        
        function v = logDet(A, op)
            %LOGDET Computation of logarithm of determinant of a matrix
            %
            %   v = logDet(A);
            %       computes the logarithm of determinant of A.
            %
            %       Here, A should be a square matrix of double or single class.
            %       If A is singular, it will return -inf.
            %
            %       Theoretically, this function should be functionally
            %       equivalent to log(det(A)). However, it avoids the
            %       overflow/underflow problems that are likely to
            %       happen when applying det to large matrices.
            %
            %       The key idea is based on the mathematical fact that
            %       the determinant of a triangular matrix equals the
            %       product of its diagonal elements. Hence, the matrix's
            %       log-determinant is equal to the sum of their logarithm
            %       values. By keeping all computations in log-scale, the
            %       problem of underflow/overflow caused by product of
            %       many scalars can be effectively circumvented.
            %
            %       The implementation is based on LU factorization.
            %
            %   v = logDet(A, 'chol');
            %       If A is positive definite, you can tell the function
            %       to use Cholesky factorization to accomplish the task
            %       using this syntax, which is substantially more efficient
            %       for positive definite matrix.
            %
            %   Remarks
            %   -------
            %       logarithm of determinant of a matrix widely occurs in the
            %       context of multivariate statistics. The log-pdf, entropy,
            %       and divergence of Gaussian distribution typically comprises
            %       a term in form of log-determinant. This function might be
            %       useful there, especially in a high-dimensional space.
            %
            %       Theoretially, LU, QR can both do the job. However, LU
            %       factorization is substantially faster. So, for generic
            %       matrix, LU factorization is adopted.
            %
            %       For positive definite matrices, such as covariance matrices,
            %       Cholesky factorization is typically more efficient. And it
            %       is strongly recommended that you use the chol (2nd syntax above)
            %       when you are sure that you are dealing with a positive definite
            %       matrix.
            %
            %   Examples
            %   --------
            %       % compute the log-determinant of a generic matrix
            %       A = rand(1000);
            %       v = logDet(A);
            %
            %       % compute the log-determinant of a positive-definite matrix
            %       A = rand(1000);
            %       C = A * A';     % this makes C positive definite
            %       v = logDet(C, 'chol');
            %
            
            %   Copyright 2008, Dahua Lin, MIT
            %   Email: dhlin@mit.edu
            %
            %   This file can be freely modified or distributed for any kind of
            %   purposes.
            %
            
            % argument checking
            assert(isfloat(A) && ndims(A) == 2 && size(A,1) == size(A,2), ...
                'logDet:invalidarg', ...
                'A should be a square matrix of double or single class.');
            if nargin < 2
                useChol = 0;
            else
                assert(strcmpi(op, 'chol'), ...
                    'logDet:invalidarg', ...
                    'The second argument can only be a string ''chol'' if it is specified.');
                useChol = 1;
            end
            
            % computation
            if useChol
                try
                    v = 2*sum(log(diag(chol(A))));
                catch
                    disp('Matrix not Pos Def')
                    keyboard
                end
            else
                [~, U, P] = lu(A);
                du = diag(U);
                c = det(P)*prod(sign(du));
                v = log(c) + sum(log(abs(du)));
            end
            if (isnan(v)|| v == -Inf || v == Inf)
                keyboard
            end
        end


        function [J, VSmooth, VVSmooth] = offlineSmoother(A, Gamma, VFilt, varargin)
            % INPUTS:
            % A - the system matrix
            % Gamma - the process noise matrix
            % VFilt - filtered variances
            
            % OPTIONAL INPUTS (string/value pairs [default in brackets])
            % 'model' - allows for a time-varying model wherein A=A(t).
            %     model(t)=m means use params from model m at time t [ones(1,T)]
            %     In this case, matrix A takes an additional final dimension,
            %     i.e., A(:,:,m).
            %
            % OUTPUTS (where X is the hidden state being estimated)
            % VSmooth(:,:,t) = Cov[X(:,t) | y(:,1:T)]
            % VVSmooth(:,:,t) = E[X(:,t-1)x(:,t)' | y(:,1:T)], t>1
            % J - See Bishop 'Pattern recognition and machine learning' p. 641
            
            % Hyperparameters and default initialization
            ss = size(A,1); % size of state space
            T = size(VFilt,3);
            % set default params
            model = ones(1,T);
            % Parse optional arguments
            args = varargin;
            nargs = length(args);
            for i=1:2:nargs
                switch args{i}
                    case 'model', model = args{i+1};
                    otherwise, error(['unrecognized argument ' args{i}])
                end
            end
            % Compute smoothed probabilities
            J = zeros(ss,ss,T);
            VSmooth = zeros(ss,ss,T);
            VVSmooth = zeros(ss,ss,T); % VV(:,:,t) = Cov[X(:,t), X(:,t-1) | y(:,1:T)] t >= 2
            VSmooth(:,:,T) = VFilt(:,:,T);
            for t = T-1:-1:1
                m = model(t+1);
                A_ = A(:,:,m);
                P = A_*VFilt(:,:,t)*A_' + Gamma;
                J(:,:,t) = VFilt(:,:,t)*A_'/P;
                VSmooth(:,:,t) = VFilt(:,:,t) + J(:,:,t)*(VSmooth(:,:,t+1) - P )*J(:,:,t)';
                VSmooth(:,:,t) = (VSmooth(:,:,t) +VSmooth(:,:,t)')/2;
                VVSmooth(:,:,t+1) = VSmooth(:,:,t+1)'*J(:,:,t)';
                VVSmooth(:,:,t+1) = (VVSmooth(:,:,t+1) + VVSmooth(:,:,t+1)')/2;
            end
        end

        
        function [xFilt] = filterOfflineToOnlineBatch(y, xInit, K, A, C, varargin)
            % Once the observations data are available, performs kalman filtering based
            % on precomputed variances and gains
            %BATCH because we compute filtered states for each class in one go,
            %       which is decisively faster.
            
            % INPUT:
            % y - observations (frames of videos)
            % xInit -expected value of latent variabvle at time 0
            % K - dimState x dimObservations x T: kalman gains, one for every timestep.
            % A - state evolution matrix
            % C - observation matrix
            %
            % OPTIONAL INPUTS (string/value pairs [default in brackets])
            % 'model' - allows for a time-varying model wherein A=A(t).
            %     model(t)=m means use params from model m at time t [ones(1,T]
            %     In this case, matrix A takes an additional final dimension,
            %     i.e., A(:,:,m).
            %
            % OUTPUT:
            % xFilt: filtered state estimates
            
            % Hyperparameters and default values
            [sizeObs, numClasses, T] = size(y); %T=numFrames
            ss = size(A,1);
            model = ones(1,T);
            % Parse optional arguments
            args = varargin;
            nargs = length(args);
            for i=1:2:nargs
                switch args{i}
                    case 'model', model = args{i+1};
                    otherwise, error(['unrecognized argument ' args{i}])
                end
            end
            
            % Filtering based on observations (hence the name 'online')
            % and kalman gain matrix
            CA = zeros(sizeObs,ss,size(A,3));   
            %CA saves us some computations if size(unique(model))<<T
            for k = unique(model)
                CA(:,:,k) = C*A(:,:,k);
            end
            xFilt = zeros(ss, numClasses, T);
            xFilt(:,:,1)=repmat(xInit, 1, numClasses) + K(:,:,1)*(y(:,:,1) - C*repmat(xInit, 1, numClasses));
            for t = 2:T
                m = model(t);
                xFilt(:,:,t) = A(:,:,m)*xFilt(:,:,t-1) + K(:,:,t)*(y(:,:,t) - CA(:,:,m)*xFilt(:,:,t-1));
            end
        end
        
        
        function EZSmooth = smootherOfflineToOnlineBatch(A, J, EZFilt, varargin)
            % Once sobservations are available, computes smoothed expected
            % values, based on filtered expected values and offline results
            % BATCH because we compute filtered states for each class in one go,
            %       which is decisively faster
            %INPUT:
            % A - state matrix
            % J - See Bishop 'Pattern recognition and machine learning' p. 641
            % E_z_filt(:,i,t) expected valued of latent variable for t-th frame of i-th
            %            video. 
            %OPTIONAL INPUT:
            % model: allows for a time-varying model wherein A=A(t). 
            %
            %OUTPUT: 
            % E_z_smooth(:,i,t) expected value for x(i,t), the latent variable
            %of the i-th video at time T. 
        
            % Hyperparameters and default value for model
            [ss, I, T] = size(EZFilt);
            model = ones(1, T);
            % parse optional input
            args = varargin;
            nargs = length(args);
            for i=1:2:nargs
              switch args{i}
               case 'model', model = args{i+1};
               otherwise, error(['unrecognized argument ' args{i}])
              end
            end
            % perform batch smoothing
            EZSmooth = zeros(ss,I,T);
            EZSmooth(:,:,T) = EZFilt(:,:,T);
            for t = T-1:-1:1
                m = model(t+1);
                EZSmooth(:,:,t) = EZFilt(:,:,t) + J(:,:,t)*(EZSmooth(:,:,t+1) - A(:,:,m)*EZFilt(:,:,t));
            end
        end

        function [loglik] = loglikOfflineToOnlineBatch(C, A, SInv, logDetS, initState, xFilt, y, varargin)
            % Once the observations data are available, computes the likelihood of the observations, based on precomputed
            % filtered expected values and innovation covariance matrices
            
            % OUTPUT:
            % loglik: log-likelihood of observed sequence
            
            % INPUT:
            % y - observations (frames of videos)
            % xFilt -expected value of latent variables (computed with KF)
            % SInv - inverse of the innovation covariance matrix
            % logDetS - logarithm of determinant of innovation covariance matrix
            % K - dimState x dimObservations x T: kalman gains, one for every timestep.
            % A - state evolution matrix
            % C - observation matrix
            
            % OPTIONAL INPUT:
            % model: allows for a time-varying model wherein A=A(t).
            
            % Hyperparameters, time for tic-toc, default model value
            %y: f x I x T
            %x_filt: Dh+Dw x I xT
            [sizeObs, numClasses, T] = size(y);
            ss = size(A,1); %size of state space
            model = ones(1,T);
            % parse optional arguments
            args = varargin;
            nargs = length(args);
            for i=1:2:nargs
                switch args{i}
                    case 'model', model = args{i+1};
                    otherwise, error(['unrecognized argument ' args{i}])
                end
            end
            % actual loglikelihood math
            CA = zeros(sizeObs, ss, size(A,3));
            for k=unique(model)
                CA(:,:,k) = C*A(:,:,k);
            end
            distFromMean = y(:,:,1) - repmat(C*initState,1,numClasses);
            quadTerm = sum(distFromMean.*(SInv(:,:,1)*distFromMean));
            loglik = -0.5*quadTerm - (sizeObs/2)*log(2*pi) - logDetS(1)/2;
            for t = 2:T
                m = model(t);
                distFromMean = (y(:,:,t) - CA(:,:,m)*xFilt(:,:,t-1));
                quadTerm = sum(distFromMean.*(SInv(:,:,t)*distFromMean));
                loglik = loglik - 0.5*quadTerm - (sizeObs/2)*log(2*pi) - logDetS(t)/2;
            end
            if(any(isnan(loglik)) || any(loglik==-Inf) || any(loglik==Inf))
                keyboard
            end
        end
 
    end
end