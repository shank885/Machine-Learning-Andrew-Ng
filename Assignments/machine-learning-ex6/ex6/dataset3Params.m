function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
increments = [ 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
incrementC = increments;
incrementSigma = increments;

best = zeros(3,1);
trialNum = 1;
for tryC = incrementC
	for tryS = incrementSigma
		fprintf(['Attempt #%d: C = %f, sigma = %f\n'], trialNum, tryC, tryS);
		trialNum = trialNum + 1;
		model = svmTrain(X, y, tryC, @(x1, x2) gaussianKernel(x1, x2, tryS));
		prediction = svmPredict(model, Xval);
		predError = mean(double(prediction == yval));
		if(predError > best(1))
			fprintf(['Best so far %f\n'], predError);
			best = [predError, tryC, tryS];
		end
	end
end

fprintf(['\n Best Found: C = %f, sigma = %f\n'], best(2), best(3));
C = best(2);
sigma = best(3);



% =========================================================================

end
