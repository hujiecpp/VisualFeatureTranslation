% Code for the methods presented in the paper 
% G. Tolias, R. Sicre and H. Jegou, Particular object retrieval with integral max-pooling of CNN activations, ICLR 2016.
% This version of the code is not optimized to run efficiently
% but to be easily readable and to reproduce the results of the paper
%
% Authored by G. Tolias, 2015. 

addpath('../utils');

% g = gpuDevice(2)
% reset(g)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if exist('aml') ~= 3
%     mex -compatibleArrayDims aml.c
% end

% % matconvnet is a prerequisite
% % run vl_setupnn for your installation to avoid downloading and compiling again
% if exist('vl_nnconv') ~= 3
%     cd matconvnet-1.0-beta25/matlab/
%     if numel(dir(fullfile('mex', 'vl_nnconv.mex*'))) == 0
%         vl_compilenn('verbose', 1, 'enableGPU', 1, 'cudaRoot', '/usr/local/cuda-8.0');
%     end
%     vl_setupnn;
%     cd ../../
% end

% net = dagnn.DagNN.loadobj(load('imagenet-resnet-101-dag.mat'));
% net.mode = 'test';
% net.conserveMemory = false;
% move(net, 'gpu')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data_dir = '../dvsd/extractD/data/';
datasets = {'Oxford5k', 'Paris6k', 'Holidays', 'Landmarks'};
desc_name = 'resnetrmac';

% number of levels in the region pyramid of R-MAC
L = 3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% extract features
fprintf('Extracting features...\n');

% Whiting
fprintf('Leaning PCA-whitening features...\n');
white_dir = [data_dir, 'Landmarks/query/'];
white_list = dir([white_dir '*.jpg']);
vecs = {};
for i = 1 : size(white_list, 1)
  img_name = [white_dir white_list(i).name];
  img = imread(img_name);
  resnetrmac = vecpostproc(rmac_regionvec(img, net, L, 0));
  % load_name = ['../sdd/dtod/data/train/Landmarks/query/', desc_name, '/', white_list(i).name, '.mat']
  % load(load_name);
  vecs{i} = resnetrmac;
  save_name = ['../sdd/dtod/data/train/Landmarks/query/', desc_name, '/', white_list(i).name, '.mat']
  save(save_name, 'resnetrmac');
end

% Learn PCA 
fprintf('Learning PCA-whitening\n');
[~, eigvec, eigval, Xm] = yael_pca (single(cell2mat(vecs)));

for i = 1 : size(datasets, 2)
    dataset = datasets{i};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if dataset(1) == 'L'
        % image files are expected under each dataset's folder %'test
        base_dir = [data_dir, dataset, '/base/'];
        base_list = dir([base_dir '*.jpg']);
        for i = 1 : size(base_list, 1)
          %%%%%%%%
          img_name = [base_dir base_list(i).name];
          img = imread(img_name);
          resnetrmac = vecpostproc(rmac_regionvec(img, net, L, 0));
          resnetrmac = vecpostproc(apply_whiten(resnetrmac, Xm, eigvec, eigval)); %
          %%%%%%%%
          save_name = ['../sdd/dtod/data/train/', dataset, '/base/', desc_name, '/', base_list(i).name, '.mat']

          %%%%%%%%
          % load(save_name);
          resnetrmac = vecpostproc(sum(resnetrmac, 2));
          %%%%%%%%

          save(save_name, 'resnetrmac');
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        % image files are expected under each dataset's folder %'test
        base_dir = [data_dir, dataset, '/base/'];
        query_dir = [data_dir, dataset, '/query/'];

        base_list = dir([base_dir '*.jpg']);
        query_list = dir([query_dir '*.jpg']);

        % Base
        for i = 1 : size(base_list, 1)
          %%%%%%%%
          img_name = [base_dir base_list(i).name];
          img = imread(img_name);
          resnetrmac = vecpostproc(rmac_regionvec(img, net, L, 0));
          resnetrmac = vecpostproc(apply_whiten(resnetrmac, Xm, eigvec, eigval)); %
          %%%%%%%%
          save_name = ['../sdd/dtod/data/test/', dataset, '/base/', desc_name, '/', base_list(i).name, '.mat']

          %%%%%%%%
          % load(save_name);
          resnetrmac = vecpostproc(sum(resnetrmac, 2));
          %%%%%%%%

          save(save_name, 'resnetrmac');
        end

        % Query
        for i = 1 : size(query_list, 1)
          %%%%%%%%
          img_name = [query_dir query_list(i).name];
          img = imread(img_name);
          resnetrmac = vecpostproc(rmac_regionvec(img, net, L, 0));
          resnetrmac = vecpostproc(apply_whiten(resnetrmac, Xm, eigvec, eigval)); %
          %%%%%%%%
          save_name = ['../sdd/dtod/data/test/', dataset, '/query/', desc_name, '/', query_list(i).name, '.mat']

          %%%%%%%%
          % load(save_name);
          resnetrmac = vecpostproc(sum(resnetrmac, 2));
          %%%%%%%%

          save(save_name, 'resnetrmac');
        end
    end
end
