clc;clear;
run('../vlfeat/toolbox/vl_setup');
% vl_version verbose

%% Set image path
% image dir - (absolute path!)
image_dir = ['/home/hujie/MAC_Retrieval/galary_images/'];
% image list
image_list = dir([image_dir '*.jpg'])
% feature save dir
save_dir = ['./SIFT/'];

%% Extract
for i = 1 : size(image_list, 1)
    img_name = [image_dir image_list(i).name];
    img = imread(img_name);
    img = single(rgb2gray(img));
    [f, sift] = vl_sift(img);
    feat_path = [save_dir image_list(i).name '.mat'];
    save(feat_path, 'sift');
    fprintf('Extracting Sift %d: %s\n', i, img_name);
    break
end