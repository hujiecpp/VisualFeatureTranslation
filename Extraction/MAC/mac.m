%
% Authored by G. Tolias, 2015. 
%
function [x, X] = MAC(I, net, flag)

	if size(I,3) == 1
		I = repmat(I, [1 1 3]);
	end

	sz = size(I);
    if sz(1) >= 1500 || sz(2) >= 1500
      I = imresize(I, [1000, 1000]);
    end
    if sz(1) < 224 || sz(2) < 224
      I = imresize(I, [224, 224]);
    end

	I = single(I) - mean(net.meta.normalization.averageImage(:));
    % I = single(ones(1000, 1000, 3));
    % sz = size(I)
	% vgg
	if flag == 1
		if ~isa(net.layers{1}.weights{1}, 'gpuArray')
			rnet = vl_simplenn(net, I);  
			X = max(rnet(end).x, 0);
		else
			rnet = vl_simplenn(net, gpuArray(I));
			X = gather(max(rnet(end).x, 0));
		end
	end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% resnet
	if flag == 0
	    I = gpuArray(I);
	    net.eval({'data', I});
	    X = gather(max(net.vars(net.getVarIndex('pool5')).value, 0));
	end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % X
    % size(X)
	x = mac_act(X);

