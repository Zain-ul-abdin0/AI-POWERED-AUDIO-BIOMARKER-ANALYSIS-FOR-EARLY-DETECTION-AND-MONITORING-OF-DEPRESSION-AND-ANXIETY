%run HOG reader example

users={'322.CLM_hog.bin'}
hog_data_dir  = '..\322_P\'


[hog_data, valid_inds, vid_id] = Read_HOG_files(users, hog_data_dir);


%save -v7.3 322.CLM_hog.mat hog_data valid_inds

