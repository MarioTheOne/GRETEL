
#./config/steel/clear-bbbp/config_bbbp_gcn-tf_clear_fold-1.json
#./config/steel/clear-bbbp/config_bbbp_gcn-tf_clear_fold-4.json
#./config/steel/clear-bbbp/config_bbbp_gcn-tf_clear_fold-5.json

for FOLD in  1 4 5
do
    qsub launch.sh main.py ./config/steel/clear-bbbp/config_bbbp_gcn-tf_clear_fold-$FOLD.json 1
done
