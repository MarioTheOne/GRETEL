
#./config/steel/clear-bbbp/config_bbbp_gcn-tf_clear_fold-3.json
#./config/steel/clear-bbbp/config_bbbp_gcn-tf_clear_fold-4.json
#./config/steel/clear-bbbp/config_bbbp_gcn-tf_clear_fold-8.json

for FOLD in  3 4 8
do
    qsub launch.sh main.py ./config/steel/clear-bbbp/config_bbbp_gcn-tf_clear_fold-$FOLD.json 1
done
