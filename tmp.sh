clear
rm tmp.txt

printf 'Start FS training\n'
python stp1_fs_train.py --perclass 5
printf 'FS training done\n'
printf 'Start 1st iter\n'
python stp2_fs_test_predict.py --load_model_path ./Params/fs_train_fs5.pth --perclass 5 --iter_idx 1
python stp3_superpixel_select.py --perclass 5 --iter_idx 1
python stp4_superpixel_train.py --save_model_path ./Params/connect_area_train_fs5_iter1.pth --perclass 5 --iter_idx 1
python stp5_test.py --load_model_path ./Params/connect_area_train_fs5_iter1.pth --perclass 5 --iter_idx 1
printf '1st iter done\n'

for i in {2..5}
do
printf "Start ${i}th iter\n"
python stp2_fs_test_predict.py --load_model_path ./Params/connect_area_train_fs5_iter${i-1}.pth --perclass 5 --iter_idx ${i}
python stp3_superpixel_select.py --perclass 5 --iter_idx ${i}
python stp4_superpixel_train.py --save_model_path ./Params/connect_area_train_fs5_iter${i}.pth --perclass 5 --iter_idx ${i}
python stp5_test.py --load_model_path ./Params/connect_area_train_fs5_iter${i}.pth --perclass 5 --iter_idx 1
printf "${i}th iter done\n"
done

