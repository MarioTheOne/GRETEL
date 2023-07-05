
#./config/steel/countergan-asd/config_asd_custom-oracle_countergan_fold-0.json
#./config/steel/countergan-asd/config_asd_custom-oracle_countergan_fold-1.json
#./config/steel/countergan-asd/config_asd_custom-oracle_countergan_fold-2.json
#./config/steel/countergan-asd/config_asd_custom-oracle_countergan_fold-3.json
#./config/steel/countergan-asd/config_asd_custom-oracle_countergan_fold-4.json
#./config/steel/countergan-asd/config_asd_custom-oracle_countergan_fold-5.json
#./config/steel/countergan-asd/config_asd_custom-oracle_countergan_fold-6.json
#./config/steel/countergan-asd/config_asd_custom-oracle_countergan_fold-7.json
#./config/steel/countergan-asd/config_asd_custom-oracle_countergan_fold-8.json
#./config/steel/countergan-asd/config_asd_custom-oracle_countergan_fold-9.json

for FOLD in  1
do
    qsub launch.sh main.py ./config/steel/countergan-asd/config_asd_custom-oracle_countergan_fold-$FOLD.json 1
done
