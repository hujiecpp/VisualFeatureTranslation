%%
clc;clear;

%%
run('./vlfeat/toolbox/vl_setup');
vl_version verbose
addpath('./utils');

datasets = {'Oxford5k', 'Paris6k', 'Holidays', 'Landmarks'};
train_num = 4000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_dir = ['../../sdd/dtod/data/train/Landmarks/query/delf/'];
train_list = dir([train_dir '*.mat']);

% Train VLAD|FV & PCA
train_delf_descriptors = {};
fprintf('Loading train delf...\n');
for i = 1 : train_num
    delf_name = [train_dir train_list(i).name];
    load(delf_name)
    train_delf_descriptors{i} = delf';%'
end
fprintf('Loading done, num: %d...\n', train_num);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Train GMM
k = 32; % number of GMMs
fprintf('Training GMM, k = %d...\n', k);
all_descriptors = [train_delf_descriptors{:}];
[means, covariances, priors] = vl_gmm(all_descriptors, k);

fprintf('Aggregation train...\n');
vecs = {};
for i = 1 : numel(train_delf_descriptors)
    % Encode using delffv     
    delffv = vecpostproc(vl_fisher(train_delf_descriptors{i}, means, covariances, priors));
    delffv = delffv';%'
    img_name = [train_dir train_list(i).name];
    fprintf('Encoding delffv %d: %s\n', i, img_name);
    fv_name = ['../../sdd/dtod/data/train/Landmarks/query/delffv/' train_list(i).name];
    save(fv_name, 'delffv');
    % load(fv_name);
    vecs{i} = delffv;
end

clear train_delf_descriptors;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Learning PCA-whitening\n');
[~, eigvec, eigval, Xm] = yael_pca(cell2mat(vecs')');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1 : size(datasets, 2)
    dataset = datasets{i};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if dataset(1) == 'L'
        base_dir = ['../../sdd/dtod/data/train/Landmarks/base/delf/'];
        base_list = dir([base_dir '*.mat']);

        % Base
        fprintf('Loading test base delf %s...\n', dataset);
        base_delf_descriptors = {};
        for i = 1 : size(base_list, 1)
            delf_name = [base_dir base_list(i).name];
            load(delf_name)
            base_delf_descriptors{i} = delf';
        end
        fprintf('Loading done, num: %d...\n', size(base_list, 1));

        % Aggre Base
        fprintf('Aggregation test base...\n');
        for i = 1 : numel(base_delf_descriptors)
            delffv = vecpostproc(vl_fisher(base_delf_descriptors{i}, means, covariances, priors));
            delffv = vecpostproc(apply_whiten(delffv, Xm, eigvec, eigval, 2048));
            delffv = delffv';
            img_name = [base_dir base_list(i).name];
            fprintf('Encoding delffv %d: %s\n', i, img_name);
            fv_name = ['../../sdd/dtod/data/train/Landmarks/base/delffv/' base_list(i).name];
            save(fv_name, 'delffv');
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        base_dir = ['../../sdd/dtod/data/test/', dataset, '/base/delf/'];
        base_list = dir([base_dir '*.mat']);

        query_dir = ['../../sdd/dtod/data/test/', dataset, '/query/delf/'];
        query_list = dir([query_dir '*.mat']);

        % Base
        fprintf('Loading test base delf %s...\n', dataset);
        base_delf_descriptors = {};
        for i = 1 : size(base_list, 1)
            delf_name = [base_dir base_list(i).name];
            load(delf_name)
            base_delf_descriptors{i} = delf';
        end
        fprintf('Loading done, num: %d...\n', size(base_list, 1));

        % Query
        fprintf('Loading test query delf %s...\n', dataset);
        query_delf_descriptors = {};
        for i = 1 : size(query_list, 1)
            delf_name = [query_dir query_list(i).name];
            load(delf_name)
            query_delf_descriptors{i} = delf';
        end
        fprintf('Loading done, num: %d...\n', size(query_list, 1));

        % Aggre Base
        fprintf('Aggregation test base...\n');
        for i = 1 : numel(base_delf_descriptors)   
            delffv = vecpostproc(vl_fisher(base_delf_descriptors{i}, means, covariances, priors));
            delffv = vecpostproc(apply_whiten (delffv, Xm, eigvec, eigval, 2048));
            delffv = delffv';
            img_name = [base_dir base_list(i).name];
            fprintf('Encoding delffv %d: %s\n', i, img_name);
            fv_name = ['../../sdd/dtod/data/test/', dataset, '/base/delffv/' base_list(i).name];
            save(fv_name, 'delffv');
        end

        % Aggre Query
        fprintf('Aggregation test query...\n');
        for i = 1 : numel(query_delf_descriptors)
            delffv = vecpostproc(vl_fisher(query_delf_descriptors{i}, means, covariances, priors));
            delffv = vecpostproc(apply_whiten (delffv, Xm, eigvec, eigval, 2048));
            delffv = delffv';
            img_name = [query_dir query_list(i).name];
            fprintf('Encoding delffv %d: %s\n', i, img_name);
            fv_name = ['../../sdd/dtod/data/test/', dataset, '/query/delffv/' query_list(i).name];
            save(fv_name, 'delffv');
        end
        clear base_delf_descriptors, query_delf_descriptors;
    end
end

