%Question 2

%I know the real labels of the data are + and -, but it is easier to run
%the code with  specific numerical values for the label, which is why - is
%referred to as 0 and + is referred to as 1 in this code. The results do
%not change because of this.

%Clearing memory
clear all

%Setting seed to ensure reproducibility
rng(1881)

%Section 1
%Number of feature dimensions
n = 2; 
%Number of iid samples
N = 999; 
%Means and variances
mu(:,1) = [-2;4]; 
mu(:,2) = [3;3];
Sigma(:,:,1) = [10 -4;-4 3]; 
Sigma(:,:,2) = [5 2;2 4];
%Class priors for labels - and + respectively
p = [0.3,0.7]; 
label = rand(1,N) >= p(1);
%Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))]; 
%Save up space
x = zeros(n,N); 
%Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

%Plotting the actual data
figure(1), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class -','Class +'), 
title('Data and Their True Labels'),
xlabel('x_1'), ylabel('x_2'),

%Checking whether eigenvalues are unique (which they are)
[V1,D1] = eig(Sigma(:,:,1));
[V2,D2] = eig(Sigma(:,:,2));

%Section 2

%Fisher LDA Error Minimization
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
%LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
[V,D] = eig(inv(Sw)*Sb); 
[~,ind] = sort(diag(D),'descend');
%Fisher LDA projection vector
wLDA = V(:,ind(1)); 
%All data projected on to the line spanned by wLDA
yLDA = wLDA'*x; 
%Ensuring class 1 falls on the + side of the axis
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; 
%Flipping yLDA accordingly
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; 
figure(2), clf,
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'ob'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+m'), axis equal,
legend('Class -','Class +'), 
title('LDA Projection of Data with Real Population/Class Labels'),
xlabel('yLDA-bLDA'),

%Finding the threshold (I will just get the best number, not the median of the two best)
for i = 1:N
%Assuming threshold is yLDA
decision(:,i) = (yLDA >= yLDA(i));
dec = transpose(decision(:,i));
%Probability of true negative
ind00 = find(dec==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(dec==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(dec==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(dec==1 & label==1); p11 = length(ind11)/Nc(2); 
%Number of errors
error(i) = length(ind10)+length(ind01);
%Probability of errors
proberror(i) = error(i)/N;
clear ind00 ind01 ind10 ind11 p00 p01 p10 p11 dec
end

%Finding the row number of the minimum error producing threshold choice
[t1,t2] = min(error);
disp('Probability of error for LDA')
disp(error(t2))
disp(proberror(t2))

%Defining bLDA
bLDA = yLDA(t2);
yLDA = yLDA-bLDA;
bLDA = -bLDA;

%Decisions of the error minimizing threshold
dec = transpose(decision(:,t2));
%Probability of true negative
ind00 = find(dec==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(dec==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(dec==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(dec==1 & label==1); p11 = length(ind11)/Nc(2); 


%Class 0 gets a circle, class 1 gets a  +, correct green, incorrect red
figure(3), 
plot(yLDA(ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(ind01),zeros(1,length(ind01)),'+r'); hold on,
plot(yLDA(ind11),zeros(1,length(ind11)),'+g'); hold on,
%Threshold is the yLDA at the error minimizing point
xline(yLDA(t2))
axis equal,
legend('Correct decisions for data from Class -','Wrong decisions for data from Class -','Wrong decisions for data from Class +','Correct decisions for data from Class +','Model threshold=0' ), 
title('Projected Data with Real Population/Class Labels and LDA Decisions'),
title('Data and Their True Labels'),
xlabel('yLDA'),

figure(4), 
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,
legend('Correct decisions for data from Class -','Wrong decisions for data from Class -','Wrong decisions for data from Class +','Correct decisions for data from Class +'), 
title('Data with Real Population/Class Labels and LDA Decisions'),
xlabel('x_1'), ylabel('x_2'),

%Removing variables
clear ind00 ind01 ind10 ind11

%Section 3

%Finding the minimums
initial = [wLDA(1) wLDA(2) bLDA];
fun1 = @(w,s)(-1).*(1/N).*(sum(log(1./(1+exp([w(1) w(2)]*s(:,label==1)+w(3)))))+sum(log(1-(1./(1+exp([w(1) w(2)]*s(:,label==0)+w(3)))))));
s = x;
fun2 = @(w)fun1(w,s);
logistic = fminsearch(fun2, initial);

%Logistic classifier
y = 1./(1+exp([logistic(1) logistic(2)]*x+logistic(3)));
labellog = zeros(1,N);
for i = 1:N
    if y(i) < 0.5
        labellog(i) = 0;
    else
        labellog(i) = 1;
    end
end

%Probability of true negative
ind00 = find(labellog==0 & label==0); 
%Probability of false positive
ind10 = find(labellog==1 & label==0); 
%Probability of false negative
ind01 = find(labellog==0 & label==1); 
%Probability of true positive
ind11 = find(labellog==1 & label==1);  
%Number of errors
errorlog = length(ind10)+length(ind01);
%Probability of errors
proberrorlog = errorlog/N;
labelDiff = label - labellog;

disp('Probability of error for logistic function')
disp(errorlog)
disp(proberrorlog)

figure(5);
plot(labelDiff);
title('Difference Between Actual Labels and Logistic Labels'),
xlabel('N'), ylabel('Difference'),

figure(6), 
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,
legend('Correct decisions for data from Class -','Wrong decisions for data from Class -','Wrong decisions for data from Class +','Correct decisions for data from Class +'), 
title('Data with Real Population/Class Labels and Logistic Decisions'),
xlabel('x_1'), ylabel('x_2'),

%Removing variables
clear ind00 ind01 ind10 ind11

%Section 4

%I am confused about the MAP classifier for this section, so I am including
%a MAP estimator as well, just in case

%Loss values (0-1 for this error minimization)
lambda = [0 1;1 0]; 
%Threshold
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); 
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));
decision = (discriminantScore >= log(gamma));

%Probability of true negative
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); 
%Number of errors
errorMAP = p10*Nc(1)+p01*Nc(2);
%Probability of errors
proberrorMAP = [p10,p01]*Nc'/N;

%Class - gets a circle, class + gets a  +, correct green, incorrect red
figure(7), 
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,
legend('Correct decisions for data from Class -','Wrong decisions for data from Class -','Wrong decisions for data from Class +','Correct decisions for data from Class +'), 
title('Data with Real Population/Class Labels and MAP Decisions'),
xlabel('x_1'), ylabel('x_2'), 

disp('Probability of error for MAP')
disp(errorMAP)
disp(proberrorMAP)

function g = evalGaussian(x,mu,Sigma)
%Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end