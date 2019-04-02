%%
clc;clear;

%%
run('../vlfeat/toolbox/vl_setup');
vl_version verbose
addpath('../utils');

datasets = {'Oxford5k', 'Paris6k', 'Holidays', 'Landmarks'};
train_num = 4000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_dir = ['../../sdd/dtod/data/train/Landmarks/query/sift/'];
train_list = dir([train_dir '*.mat']);

% Train VLAD|FV & PCA
train_sift_descriptors = {};
fprintf('Loading train sift...\n');
for i = 1 : train_num
    sift_name = [train_dir train_list(i).name];
    load(sift_name)

    sift = vecs_normalize(sift, 1);
    sift = sift .^ (1.0/2);

    train_sift_descriptors{i} = sift;
end
fprintf('Loading done, num: %d...\n', train_num);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Train GMM
k = 32; % number of GMMs
fprintf('Training GMM, k = %d...\n', k);
all_descriptors = [train_sift_descriptors{:}];
[means, covariances, priors] = vl_gmm(all_descriptors, k);

fprintf('Aggregation train...\n');
vecs = {};
for i = 1 : numel(train_sift_descriptors)
    % Encode using siftfv     
    siftfv = vecpostproc(vl_fisher(train_sift_descriptors{i}, means, covariances, priors));
    siftfv = siftfv';%'
    img_name = [train_dir train_list(i).name];
    fprintf('Encoding siftfv %d: %s\n', i, img_name);
    fv_name = ['../../sdd/dtod/data/train/Landmarks/query/siftfv/' train_list(i).name];
    save(fv_name, 'siftfv');
    % load(fv_name);
    vecs{i} = siftfv;
end

clear train_sift_descriptors;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Learning PCA-whitening\n');
[~, eigvec, eigval, Xm] = yael_pca(cell2mat(vecs')');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1 : size(datasets, 2)
    dataset = datasets{i};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if dataset(1) == 'L'
        base_dir = ['../../sdd/dtod/data/train/Landmarks/base/sift/'];
        base_list = dir([base_dir '*.mat']);

        % Base
        fprintf('Aggre test base sift %s...\n', dataset);
        for i = 1 : size(base_list, 1)
            sift_name = [base_dir base_list(i).name];
            load(sift_name)

            sift = vecs_normalize(sift, 1);
            sift = sift .^ (1.0/2);

            % Aggre Base
            siftfv = vecpostproc(vl_fisher(sift, means, covariances, priors));
            siftfv = vecpostproc(apply_whiten(siftfv, Xm, eigvec, eigval, 2048));
            siftfv = siftfv';
            img_name = [base_dir base_list(i).name];
            fprintf('Encoding siftfv %d: %s\n', i, img_name);
            fv_name = ['../../sdd/dtod/data/train/Landmarks/base/siftfv/' base_list(i).name];
            save(fv_name, 'siftfv');
        end
        fprintf('Aggre base done, num: %d...\n', size(base_list, 1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        base_dir = ['../../sdd/dtod/data/test/', dataset, '/base/sift/'];
        base_list = dir([base_dir '*.mat']);

        query_dir = ['../../sdd/dtod/data/test/', dataset, '/query/sift/'];
        query_list = dir([query_dir '*.mat']);

        % Base
        fprintf('Aggre test base sift %s...\n', dataset);
        for i = 1 : size(base_list, 1)
            sift_name = [base_dir base_list(i).name];
            load(sift_name)

            sift = vecs_normalize(sift, 1);
            sift = sift .^ (1.0/2);

            % Aggre Base
            siftfv = vecpostproc(vl_fisher(sift, means, covariances, priors));
            siftfv = vecpostproc(apply_whiten (siftfv, Xm, eigvec, eigval, 2048));
            siftfv = siftfv';
            img_name = [base_dir base_list(i).name];
            fprintf('Encoding siftfv %d: %s\n', i, img_name);
            fv_name = ['../../sdd/dtod/data/test/', dataset, '/base/siftfv/' base_list(i).name];
            save(fv_name, 'siftfv');
        end
        fprintf('Aggre test base done, num: %d...\n', size(base_list, 1));

        % Query
        fprintf('Aggre test query sift %s...\n', dataset);
        for i = 1 : size(query_list, 1)
            sift_name = [query_dir query_list(i).name];
            load(sift_name)

            sift = vecs_normalize(sift, 1);
            sift = sift .^ (1.0/2);

            % Aggre Query
            siftfv = vecpostproc(vl_fisher(sift, means, covariances, priors));
            siftfv = vecpostproc(apply_whiten (siftfv, Xm, eigvec, eigval, 2048));
            siftfv = siftfv';
            img_name = [query_dir query_list(i).name];
            fprintf('Encoding siftfv %d: %s\n', i, img_name);
            fv_name = ['../../sdd/dtod/data/test/', dataset, '/query/siftfv/' query_list(i).name];
            save(fv_name, 'siftfv');
        end
        fprintf('Aggre test base done, num: %d...\n', size(query_list, 1));
    end
end
