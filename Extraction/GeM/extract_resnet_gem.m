[root] = fileparts(mfilename('fullpath')) ;
% add paths from this package
addpath(fullfile(root, 'cnnblocks'));
addpath(fullfile(root, 'cnninit'));
addpath(fullfile(root, 'cnntrain'));
addpath(fullfile(root, 'cnnvecs'));
addpath(fullfile(root, 'examples'));
addpath(fullfile(root, 'whiten')); 
addpath(fullfile(root, 'utils')); 
addpath(fullfile(root, 'yael')); 
addpath(fullfile(root, 'helpers')); 

g = gpuDevice(4)
reset(g)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% matconvnet is a prerequisite
% run vl_setupnn for your installation to avoid downloading and compiling again
if exist('vl_nnconv') ~= 3
    cd matconvnet-1.0-beta25/matlab/
    if numel(dir(fullfile('mex', 'vl_nnconv.mex*'))) == 0
        vl_compilenn('verbose', 1, 'enableGPU', 1, 'cudaRoot', '/usr/local/cuda-8.0');
    end
    vl_setupnn;
    cd ../../
end

use_ms = 1; % use multi-scale representation, otherwise use single-scale
use_rvec = 0;  % use regional representation (R-MAC, R-GeM), otherwise use global (MAC, GeM)
use_gpu = [1];  % use GPUs (array of GPUIDs), if empty use CPU

network_file = './data/retrievalSfM120k-gem-resnet101.mat';

% Prepare function for desc extraction
if ~use_rvec 
    if ~use_ms
        descfun = @(x, y) cnn_vecms (x, y, 1);
    else
        descfun = @(x, y) cnn_vecms (x, y, [1, 1/sqrt(2), 1/2]);
    end  
else 
    if ~use_ms
        descfun = @(x, y) cnn_vecrms (x, y, 3, 1);
    else
        descfun = @(x, y) cnn_vecrms (x, y, 3, [1, 1/sqrt(2), 1/2]);
    end  
end

[~, network_name, ~] = fileparts(network_file);
fprintf('>> %s: Evaluating CNN image retrieval...\n', network_name);

load(network_file);
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';
move(net, 'gpu')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data_dir = '../dvsd/extractD/data/';
datasets = {'Oxford5k', 'Paris6k', 'Holidays', 'Landmarks'};
desc_name = 'resnetgem';

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
  resnetgem = vecpostproc(descfun(img, net));
  % load_name = ['../sdd/dtod/data/train/Landmarks/query/', desc_name, '/', white_list(i).name, '.mat']
  % load(load_name);
  vecs{i} = resnetgem;
  save_name = ['../sdd/dtod/data/train/Landmarks/query/', desc_name, '/', white_list(i).name, '.mat']
  save(save_name, 'resnetgem');
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
          img_name = [base_dir base_list(i).name];
          img = imread(img_name);
          resnetgem = vecpostproc(descfun(img, net));
          resnetgem = vecpostproc(apply_whiten(resnetgem, Xm, eigvec, eigval)); %
          save_name = ['../sdd/dtod/data/train/', dataset, '/base/', desc_name, '/', base_list(i).name, '.mat']
          save(save_name, 'resnetgem');
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
          img_name = [base_dir base_list(i).name];
          img = imread(img_name);
          resnetgem = vecpostproc(descfun(img, net));
          resnetgem = vecpostproc(apply_whiten(resnetgem, Xm, eigvec, eigval)); %
          save_name = ['../sdd/dtod/data/test/', dataset, '/base/', desc_name, '/', base_list(i).name, '.mat']
          save(save_name, 'resnetgem');
        end

        % Query
        for i = 1 : size(query_list, 1)
          img_name = [query_dir query_list(i).name];
          img = imread(img_name);
          resnetgem = vecpostproc(descfun(img, net));
          resnetgem = vecpostproc(apply_whiten(resnetgem, Xm, eigvec, eigval)); %
          save_name = ['../sdd/dtod/data/test/', dataset, '/query/', desc_name, '/', query_list(i).name, '.mat']
          save(save_name, 'resnetgem');
        end
    end
end