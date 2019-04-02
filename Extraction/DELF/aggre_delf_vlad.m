%%
clc;clear;

%%
run('../vlfeat/toolbox/vl_setup');
vl_version verbose
addpath('../utils');

datasets = {'Landmarks', 'Holidays', 'Paris6k', 'Oxford5k'};
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

%% Train K-MEANS
k = 64;
fprintf('Training K-MEANS, k = %d...\n', k);
all_descriptors = [train_delf_descriptors{:}];
centroids = vl_kmeans(all_descriptors, k);
kdtree = vl_kdtreebuild(centroids);

fprintf('Aggregation train...\n');
vecs = {};
for i = 1 : numel(train_delf_descriptors)
    nn = vl_kdtreequery(kdtree, centroids, train_delf_descriptors{i});
    assignments = zeros(k, numel(nn));
    assignments(sub2ind(size(assignments), nn, 1:numel(nn))) = 1;

    delfvlad = vecpostproc(vl_vlad(train_delf_descriptors{i}, centroids, assignments));
    delfvlad = delfvlad';%'

    img_name = [train_dir train_list(i).name];
    fprintf('Encoding delfvlad %d: %s\n', i, img_name);
    vlad_name = ['../../sdd/dtod/data/train/Landmarks/query/delfvlad/' train_list(i).name];
    save(vlad_name, 'delfvlad');
    % load(vlad_name);
    vecs{i} = delfvlad;
end

clear train_delf_descriptors;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Learning PCA-whitening\n');
[~, eigvec, eigval, Xm] = yael_pca(single(cell2mat(vecs')'));%

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
            nn = vl_kdtreequery(kdtree, centroids, base_delf_descriptors{i});
            assignments = zeros(k, numel(nn));
            assignments(sub2ind(size(assignments), nn, 1:numel(nn))) = 1;

            delfvlad = vecpostproc(vl_vlad(base_delf_descriptors{i}, centroids, assignments));
            delfvlad = vecpostproc(apply_whiten(delfvlad, Xm, eigvec, eigval, 2048)); %
            delfvlad = delfvlad';%'

            img_name = [base_dir base_list(i).name];
            fprintf('Encoding delfvlad %d: %s\n', i, img_name);
            vlad_name = ['../../sdd/dtod/data/train/Landmarks/base/delfvlad/' base_list(i).name];
            save(vlad_name, 'delfvlad');
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
            nn = vl_kdtreequery(kdtree, centroids, base_delf_descriptors{i});
            assignments = zeros(k, numel(nn));
            assignments(sub2ind(size(assignments), nn, 1:numel(nn))) = 1;

            delfvlad = vecpostproc(vl_vlad(base_delf_descriptors{i}, centroids, assignments));
            delfvlad = vecpostproc(apply_whiten(delfvlad, Xm, eigvec, eigval, 2048)); %
            delfvlad = delfvlad';%'

            img_name = [base_dir base_list(i).name];
            fprintf('Encoding delfvlad %d: %s\n', i, img_name);
            vlad_name = ['../../sdd/dtod/data/test/', dataset, '/base/delfvlad/' base_list(i).name];
            save(vlad_name, 'delfvlad');
        end

        % Aggre Query
        fprintf('Aggregation test query...\n');
        for i = 1 : numel(query_delf_descriptors)
            nn = vl_kdtreequery(kdtree, centroids, query_delf_descriptors{i});
            assignments = zeros(k, numel(nn));
            assignments(sub2ind(size(assignments), nn, 1:numel(nn))) = 1;

            delfvlad = vecpostproc(vl_vlad(query_delf_descriptors{i}, centroids, assignments));
            delfvlad = vecpostproc(apply_whiten(delfvlad, Xm, eigvec, eigval, 2048)); %
            delfvlad = delfvlad';%'

            img_name = [query_dir query_list(i).name];
            fprintf('Encoding delfvlad %d: %s\n', i, img_name);
            vlad_name = ['../../sdd/dtod/data/test/', dataset, '/query/delfvlad/' query_list(i).name];
            save(vlad_name, 'delfvlad');
        end

        clear base_delf_descriptors, query_delf_descriptors;
        
    end
end

