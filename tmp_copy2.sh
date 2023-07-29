for j in {1..2}
do
  printf "epo ${j} Start FS training\n"
  python stp1_fs_train.py --perclass 5 --epo ${j}
  printf 'FS training done\n'
  printf "epo ${j} Start 1st iter\n"
  python stp2_fs_test_predict.py --load_model_path ./Params/fs_train_fs5.pth --perclass 5 --iter_idx 1 --epo ${j}

  python stp3_superpixel_selectwpseudolabel.py --perclass 5 --iter_idx 1 --epo ${j}

  python stp4_superpixel_train.py --load_model_path ./Params/fs_train_fs5.pth --save_model_path ./Params/connect_area_train_fs5_iter1.pth --perclass 5 --iter_idx 1 --epo ${j}

  python stp5_test.py --load_model_path ./Params/connect_area_train_fs5_iter1.pth --perclass 5 --iter_idx 1 --epo ${j}
  printf '1st iter done\n'

  for i in {2..10}
  do
    printf "Start epo ${j} iter ${i}th \n"
    python stp2_fs_test_predict.py --load_model_path ./Params/connect_area_train_fs5_iter$(($i - 1)).pth --perclass 5 --iter_idx ${i} --epo ${j}

    python stp3_superpixel_selectwpseudolabel.py --perclass 5 --iter_idx ${i} --epo ${j}

    python stp4_superpixel_train.py --load_model_path ./Params/connect_area_train_fs5_iter$(($i - 1)).pth  --save_model_path ./Params/connect_area_train_fs5_iter${i}.pth --perclass 5 --iter_idx ${i} --epo ${j}

    python stp5_test.py --load_model_path ./Params/connect_area_train_fs5_iter${i}.pth --perclass 5 --iter_idx  ${i} --epo ${j}
    printf "epo ${j} iter ${i}th  done\n"
  printf ' '
  done
done

