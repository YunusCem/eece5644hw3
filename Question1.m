%Question 1

%Setting Seed (to make my work replicable)
rng(1881)

%Clearing memory
clear all

%Counting variables to ensure the program is not stuck
countN = 0;

%All four sample sizes
for i = 1:4
    
    countN = countN+1
    
    %Tolerance for EM stopping criterion
    delta = 1e-4;
    %Regularization parameter for covariance estimates
    regWeight = 1e-10; 
    %K-Fold Cross Validation
    K = 10; 
    %Number of samples
    N = [10,100,1000,10000];

    %Generate samples from a 4-component GMM
    alpha_true = [0.17,0.22,0.28,0.33];
    mu_true = [10 -10 -10 10;-10 10 -10 10];
    Sigma_true(:,:,1) = [25 1;1 20];
    Sigma_true(:,:,2) = [27 4;4 5];
    Sigma_true(:,:,3) = [15 -9;-9 15];
    Sigma_true(:,:,4) = [4 1;1 22];
    x = randGMM(N(i),alpha_true,mu_true,Sigma_true);
    
    %Plotting data
    figure(i), clf,
    plot(x(1,:),x(2,:),'ob')
    xlabel('x1'); ylabel('x2');
    title(strcat('Data with N=',num2str(N(i))));
    
    %To determine dimensionality of samples and number of GMM components
    [d,MM] = size(mu_true); 

    %Divide the data set into 10 approximately-equal-sized partitions
    dummy = ceil(linspace(0,N(i),K+1));
    for k = 1:K
        indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
    end

    %Allocate space
    loglikelihoodtrain = zeros(K,6); loglikelihoodvalidate = zeros(K,6); 
    Averagelltrain = zeros(1,6); Averagellvalidate = zeros(1,6);

    countM = 0;

    %Try all 6 mixture options
    for M = 1:6
        
        countM = countM+1
        countk = 0;

        %10-fold cross validation
        for k = 1:K
            countk = countk+1
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            %Using folk k as validation set
            x1Validate = x(1,indValidate); 
            x2Validate = x(2,indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N(i)];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,2):N(i)];
            end
            
            %Using all other folds as training set
            x1Train = x(1,indTrain); 
            x2Train = x(2,indTrain);
            xTrain = [x1Train; x2Train];
            xValidate = [x1Validate; x2Validate];
            Ntrain = length(indTrain); Nvalidate = length(indValidate);
            
            %Train model parameters (EM)
            %Initialize the GMM to randomly selected samples
            alpha = ones(1,M)/M;
            shuffledIndices = randperm(Ntrain);
            %Pick M random samples as initial mean estimates (this led
            %to good initial estimates (better log likelihoods))
            mu = xTrain(:,shuffledIndices(1:M)); 
            %Assign each sample to the nearest mean (better initialization)
            [~,assignedCentroidLabels] = min(pdist2(mu',xTrain'),[],1); 
            %Use sample covariances of initial assignments as initial covariance estimates
            for m = 1:M 
                Sigma(:,:,m) = cov(xTrain(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
            end
            t = 0;
            
            %Not converged at the beginning
            Converged = 0; 

            while ~Converged
                for l = 1:M
                    temp(l,:) = repmat(alpha(l),1,Ntrain).*evalGaussian(xTrain,mu(:,l),Sigma(:,:,l));
                end
                plgivenx = temp./sum(temp,1);
                clear temp
                alphaNew = mean(plgivenx,2);
                w = plgivenx./repmat(sum(plgivenx,2),1,Ntrain);
                muNew = xTrain*w';
                for l = 1:M
                    v = xTrain-repmat(muNew(:,l),1,Ntrain);
                    u = repmat(w(l,:),d,1).*v;
                    %Adding a small regularization term
                    SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); 
                end
                Dalpha = sum(abs(alphaNew-alpha));
                Dmu = sum(sum(abs(muNew-mu)));
                DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
                %Check if converged
                Converged = ((Dalpha+Dmu+DSigma)<delta); 
                alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
                t = t+1;
            end
            %Validation
            loglikelihoodtrain(k,M) = sum(log(evalGMM(xTrain,alpha,mu,Sigma)));
            loglikelihoodvalidate(k,M) = sum(log(evalGMM(xValidate,alpha,mu,Sigma)));
           
        end
        
        %Average Performance Variables
        Averagelltrain(1,M) = mean(loglikelihoodtrain(:,M)); 
        BICtrain(1,M) = -2*Averagelltrain(1,M)+M*log(N(i));
        Averagellvalidate(1,M) = mean(loglikelihoodvalidate(:,M)); 
        %Sometimes the log likelihoods for N=10 are zero, leading to
        %negative infinity results. I assume that this is instead the
        %lowest log likelihood value instead (so it is possible to graph).
        if isinf(Averagellvalidate(1,M))
            Averagellvalidate(1,M) = (min(Averagellvalidate(find(isfinite(Averagellvalidate)))));
        end
        BICvalidate(1,M) = -2*Averagellvalidate(1,M)+M*log(N(i));
        %Recording values
        TotBICValidate(i,M) = BICvalidate(1,M);
        TotBICTrain(i,M) = BICtrain(1,M);
        TotAvgllValidate(i,M) = Averagellvalidate(1,M);
        TotAvgllTrain(i,M) = Averagelltrain(1,M);
    end

%Recording Best Outcomes
[LowestBIC orderB] = min(BICvalidate)
[Lowestll orderl] = max(Averagellvalidate)

figure(i+4), clf,
plot(Averagelltrain,'.b'); 
xlabel('GMM Number'); ylabel(strcat('Log likelihood estimate with ',num2str(K),'-fold cross-validation'));
title(strcat('Training Log-Likelihoods for N=',num2str(N(i))));
grid on

figure(i+8), clf,
plot(Averagellvalidate,'rx');
xlabel('GMM Number'); ylabel(strcat('Log likelihood estimate with ',num2str(K),'-fold cross-validation'));
title(strcat('Validation Log-Likelihoods for N=',num2str(N(i))));
grid on

figure(i+12), clf,
plot(BICtrain,'.b');
xlabel('GMM Number'); ylabel(strcat('BIC estimate with ',num2str(K),'-fold cross-validation'));
title(strcat('Training BICs for N=',num2str(N(i))));
grid on

figure(i+16), clf,
plot(BICvalidate,'rx');
xlabel('GMM Number'); ylabel(strcat('BIC estimate with ',num2str(K),'-fold cross-validation'));
title(strcat('Validation BICs for N=',num2str(N(i))))
grid on

%Saving values
BICorder(i) = orderB;
BIClow(i) = LowestBIC;
lorder(i) = orderl;
lllow(i) = Lowestll;

end

%Print the important values
BICorder
BIClow
lorder
lllow
TotBICValidate
TotBICTrain
TotAvgllValidate
TotAvgllTrain

%%%
function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end

function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end