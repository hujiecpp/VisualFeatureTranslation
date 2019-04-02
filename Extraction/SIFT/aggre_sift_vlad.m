%%
clc;clear;

%%
run('./vlfeat/toolbox/vl_setup');
vl_version verbose
addpath('./utils');

datasets = {'Landmarks', 'Holidays', 'Paris6k', 'Oxford5k'};
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

%% Train K-MEANS
k = 64;
fprintf('Training K-MEANS, k = %d...\n', k);
all_descriptors = [train_sift_descriptors{:}];
centroids = vl_kmeans(all_descriptors, k);
kdtree = vl_kdtreebuild(centroids);

fprintf('Aggregation train...\n');
vecs = {};
for i = 1 : numel(train_sift_descriptors)
    nn = vl_kdtreequery(kdtree, centroids, train_sift_descriptors{i});
    assignments = zeros(k, numel(nn));
    assignments(sub2ind(size(assignments), nn, 1:numel(nn))) = 1;

    siftvlad = vecpostproc(vl_vlad(train_sift_descriptors{i}, centroids, assignments));
    siftvlad = siftvlad';%'

    img_name = [train_dir train_list(i).name];
    fprintf('Encoding siftvlad %d: %s\n', i, img_name);
    vlad_name = ['../../sdd/dtod/data/train/Landmarks/query/siftvlad/' train_list(i).name];
    save(vlad_name, 'siftvlad');
    % load(vlad_name);
    vecs{i} = siftvlad;
end

clear train_sift_descriptors;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Learning PCA-whitening\n');
[~, eigvec, eigval, Xm] = yael_pca(single(cell2mat(vecs')'));%

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
            nn = vl_kdtreequery(kdtree, centroids, sift);
            assignments = zeros(k, numel(nn));
            assignments(sub2ind(size(assignments), nn, 1:numel(nn))) = 1;

            siftvlad = vecpostproc(vl_vlad(sift, centroids, assignments));
            siftvlad = vecpostproc(apply_whiten(siftvlad, Xm, eigvec, eigval, 2048)); %
            siftvlad = siftvlad';%'

            img_name = [base_dir base_list(i).name];
            fprintf('Encoding siftvlad %d: %s\n', i, img_name);
            vlad_name = ['../../sdd/dtod/data/train/Landmarks/base/siftvlad/' base_list(i).name];
            save(vlad_name, 'siftvlad');

        end
        fprintf('Aggre test base done, num: %d...\n', size(base_list, 1));

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
            nn = vl_kdtreequery(kdtree, centroids, sift);
            assignments = zeros(k, numel(nn));
            assignments(sub2ind(size(assignments), nn, 1:numel(nn))) = 1;

            siftvlad = vecpostproc(vl_vlad(sift, centroids, assignments));
            siftvlad = vecpostproc(apply_whiten(siftvlad, Xm, eigvec, eigval, 2048)); %
            siftvlad = siftvlad';

            img_name = [base_dir base_list(i).name];
            fprintf('Encoding siftvlad %d: %s\n', i, img_name);
            vlad_name = ['../../sdd/dtod/data/test/', dataset, '/base/siftvlad/' base_list(i).name];
            save(vlad_name, 'siftvlad');
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
            nn = vl_kdtreequery(kdtree, centroids, sift);
            assignments = zeros(k, numel(nn));
            assignments(sub2ind(size(assignments), nn, 1:numel(nn))) = 1;

            siftvlad = vecpostproc(vl_vlad(sift, centroids, assignments));
            siftvlad = vecpostproc(apply_whiten(siftvlad, Xm, eigvec, eigval, 2048)); %
            siftvlad = siftvlad';

            img_name = [query_dir query_list(i).name];
            fprintf('Encoding siftvlad %d: %s\n', i, img_name);
            vlad_name = ['../../sdd/dtod/data/test/', dataset, '/query/siftvlad/' query_list(i).name];
            save(vlad_name, 'siftvlad');
        end
        fprintf('Aggre test query done, num: %d...\n', size(query_list, 1));

    end
end
