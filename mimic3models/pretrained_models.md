# Baselines

For each of the four main tasks we provide 7 baselines:  
* Linear/logistic regression
* Standard LSTM
* Standard LSTM + deep supervision
* Channel-wise LSTM
* Channel-wise LSTM + deep supervision
* Multitask standard LSTM
* Multitask channel-wise LSTM

Here you can find the commands for re-running the pre-trained models on the test set.
To run the models on other test sets you can change the test set path in corresponding scripts.

## In-hospital mortality

##### Logistic regression
        python -um mimic3models.in_hospital_mortality.logistic.main --l2 --C 0.001

##### Standard LSTM
        python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --depth 2 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/in_hospital_mortality/keras_states/rk_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch27.test0.278806287862.state --mode test

##### Standard LSTM + deep supervision
        python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 32 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/in_hospital_mortality/keras_states/rk_lstm.n32.d0.3.dep1.bs8.ts1.0.trc0.5.epoch25.test0.302563312449.state --mode test --target_repl_coef 0.5

##### Channel-wise LSTM
        python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/channel_wise_lstms.py --dim 8 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/in_hospital_mortality/keras_states/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state --mode test --size_coef 4.0

##### Channel-wise LSTM + deep supervision
        python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/channel_wise_lstms.py --dim 16 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/in_hospital_mortality/keras_states/rk_channel_wise_lstms.n16.szc4.0.d0.3.dep1.bs8.ts1.0.trc0.5.epoch12.test0.284083816056.state --mode test --size_coef 4.0 --target_repl_coef 0.5

##### Multitask standard LSTM
        python -um mimic3models.multitask.main --network mimic3models/keras_models/multitask_lstm.py --dim 512 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/multitask/keras_states/nrk_lstm.n512.d0.3.dep1.bs16.ts1.0.trc0.5_partition=custom_ihm=0.2_decomp=1.0_los=1.5_pheno=1.0.epoch21.test2.36226256578.state --mode test --partition custom --ihm_C 0.2 --decomp_C 1.0 --los_C 1.5 --pheno_C 1.0 --target_repl_coef 0.5

##### Multitask channel-wise LSTM
        python -um mimic3models.multitask.main --network mimic3models/keras_models/multitask_channel_wise_lstms.py --dim 16 --size_coef 8 --dropout 0.3 --depth 1 --batch_size 8 --timestep 1.0 --load_state mimic3models/multitask/keras_states/nr2k_channel_wise_lstms.n16.szc8.0.d0.3.dep1.bs64.ts1.0_partition=custom_ihm=0.2_decomp=1.0_los=1.5_pheno=1.0.epoch27.test2.39127077527.state --mode test --partition custom --ihm_C 0.2 --decomp_C 1.0 --los_C 1.5 --pheno_C 1.0

## Decompensation

##### Logistic regression
        python -um mimic3models.decompensation.logistic.main --no-grid-search

##### Standard LSTM
        python -um mimic3models.decompensation.main --network mimic3models/keras_models/lstm.py --dim 128 --depth 1 --batch_size 8 --timestep 1.0 --load_state mimic3models/decompensation/keras_states/rk_lstm.n128.dep1.bs8.ts1.0.chunk35.test0.0745138007416.state --mode test

##### Standard LSTM + deep supervision
        python -um mimic3models.decompensation.main --network mimic3models/keras_models/lstm.py --dim 128 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/decompensation/keras_states/nr2k_lstm.n128.d0.3.dep1.dsup.bs8.ts1.0.chunk49.test0.0933801653546.state --mode test --deep_supervision
        
##### Channel-wise LSTM
        python -um mimic3models.decompensation.main --network mimic3models/keras_models/channel_wise_lstms.py --dim 16 --depth 1 --batch_size 8 --timestep 1.0 --load_state mimic3models/decompensation/keras_states/nr2k_channel_wise_lstms.n16.szc4.0.dep1.bs64.ts1.0.chunk10.test0.0674870069544.state --mode test --size_coef 4.0

##### Channel-wise LSTM + deep supervision
        python -um mimic3models.decompensation.main --network mimic3models/keras_models/channel_wise_lstms.py --dim 16 --depth 1 --batch_size 8 --timestep 1.0 --load_state mimic3models/decompensation/keras_states/nrk_channel_wise_lstms.n16.szc8.0.dep1.dsup.bs32.ts1.0.chunk6.test0.0810981076094.state --mode test --size_coef 8.0 --deep_supervision

##### Multitask standard LSTM
        python -um mimic3models.multitask.main --network mimic3models/keras_models/multitask_lstm.py --dim 512 --depth 1 --batch_size 8 --dropout 0.5 --timestep 1.0 --load_state mimic3models/multitask/keras_states/nrk_lstm.n512.d0.5.dep1.bs16.ts1.0.trc0.5_partition=custom_ihm=0.2_decomp=1.0_los=1.5_pheno=1.0.epoch26.test2.35615044199.state --mode test --partition custom --ihm_C 0.2 --decomp_C 1.0 --los_C 1.5 --pheno_C 1.0 --target_repl_coef 0.5

##### Multitask channel-wise LSTM
        python -um mimic3models.multitask.main --network mimic3models/keras_models/multitask_channel_wise_lstms.py --dim 16 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/multitask/keras_states/nr2k_channel_wise_lstms.n16.szc8.0.d0.3.dep1.bs64.ts1.0.trc0.5_partition=custom_ihm=0.2_decomp=1.0_los=1.5_pheno=1.0.epoch25.test2.44558876305.state --mode test --size_coef 8.0 --partition custom --ihm_C 0.2 --decomp_C 1.0 --los_C 1.5 --pheno_C 1.0 --target_repl_coef 0.5

## Length of Stay

##### Logistic regression
        python -um mimic3models.length_of_stay.logistic.main_cf --no-grid-search

##### Standard LSTM
        python -um mimic3models.length_of_stay.main --network mimic3models/keras_models/lstm.py --dim 64 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/length_of_stay/keras_states/rk_lstm.n64.d0.3.dep1.bs8.ts1.0.partition=custom.chunk18.test1.83189093661.state --mode test --partition custom

##### Standard LSTM + deep supervision
        python -um mimic3models.length_of_stay.main --network mimic3models/keras_models/lstm.py --dim 128 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/length_of_stay/keras_states/nrk_lstm.n128.d0.3.dep1.dsup.bs8.ts1.0.partition=custom.chunk17.test1.29710156898.state --mode test --partition custom --deep_supervision
        
##### Channel-wise LSTM
        python -um mimic3models.length_of_stay.main --network mimic3models/keras_models/channel_wise_lstms.py --dim 16 --depth 1 --batch_size 8 --timestep 1.0 --load_state mimic3models/length_of_stay/keras_states/nrk_channel_wise_lstms.n16.szc8.0.dep1.bs64.ts1.0.partition=custom.chunk1.test1.81423672855.state --mode test --size_coef 8.0 --partition custom

##### Channel-wise LSTM + deep supervision
        python -um mimic3models.length_of_stay.main --network mimic3models/keras_models/channel_wise_lstms.py --dim 16 --depth 1 --batch_size 8 --timestep 1.0 --load_state mimic3models/length_of_stay/keras_states/nrk_channel_wise_lstms.n16.szc8.0.dep1.dsup.bs32.ts1.0.partition=custom.chunk6.test1.32680085614.state --mode test --size_coef 8.0 --partition custom --deep_supervision
        
##### Multitask standard LSTM
        python -um mimic3models.multitask.main --network mimic3models/keras_models/multitask_lstm.py --dim 256 --depth 1 --batch_size 8 --timestep 1.0 --load_state mimic3models/multitask/keras_states/nrk_lstm.n256.dep1.bs16.ts1.0_partition=custom_ihm=0.2_decomp=1.0_los=1.5_pheno=1.0.epoch5.test2.3837153072.state --mode test --partition custom --ihm_C 0.2 --decomp_C 1.0 --los_C 1.5 --pheno_C 1.0
        
##### Multitask channel-wise LSTM
        python -um mimic3models.multitask.main --network mimic3models/keras_models/multitask_channel_wise_lstms.py --dim 16 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/multitask/keras_states/nr2k_channel_wise_lstms.n16.szc8.0.d0.3.dep1.bs64.ts1.0.trc0.5_partition=custom_ihm=0.2_decomp=1.0_los=1.5_pheno=1.0.epoch29.test2.45277419264.state --mode test --size_coef 8.0 --partition custom --ihm_C 0.2 --decomp_C 1.0 --los_C 1.5 --pheno_C 1.0 --target_repl_coef 0.5
        
## Phenotyping

##### Logistic regression
        python -um mimic3models.phenotyping.logistic.main --no-grid-search

##### Standard LSTM
        python -um mimic3models.phenotyping.main --network mimic3models/keras_models/lstm.py --dim 256 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/phenotyping/keras_states/rk_lstm.n256.d0.3.dep1.bs8.ts1.0.epoch19.test0.34659054206.state --mode test

##### Standard LSTM + deep supervision
        python -um mimic3models.phenotyping.main --network mimic3models/keras_models/lstm.py --dim 256 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/phenotyping/keras_states/nrk_lstm.n256.d0.3.dep1.bs8.ts1.0.trc0.5.epoch18.test0.351739036659.state --mode test --target_repl_coef 0.5
        
##### Channel-wise LSTM
        python -um mimic3models.phenotyping.main --network mimic3models/keras_models/channel_wise_lstms.py --dim 16 --depth 1 --batch_size 8 --dropout 0.3 --timestep 1.0 --load_state mimic3models/phenotyping/keras_states/nr6k_channel_wise_lstms.n16.szc8.0.d0.3.dep1.bs64.ts1.0.epoch49.test0.348234337795.state --mode test --size_coef 8.0
        
##### Channel-wise LSTM + deep supervision
        python -um mimic3models.phenotyping.main --network mimic3models/keras_models/channel_wise_lstms.py --dim 16 --depth 1 --batch_size 8 --timestep 1.0 --load_state mimic3models/phenotyping/keras_states/nrk_channel_wise_lstms.n16.szc8.0.dep1.bs8.ts1.0.trc0.5.epoch12.test0.349869527944.state --mode test --size_coef 8.0 --target_repl_coef 0.5
                
##### Multitask standard LSTM
        python -um mimic3models.multitask.main --network mimic3models/keras_models/multitask_lstm.py --dim 256 --depth 1 --batch_size 8 --dropout 0.5 --timestep 1.0 --load_state mimic3models/multitask/keras_states/k_lstm.n256.d0.5.dep1.bs8.ts1.0.trc0.5_partition=custom_ihm=0.1_decomp=0.1_los=0.5_pheno=1.0.epoch31.test0.942830482632.state --mode test --partition custom --ihm_C 0.1 --decomp_C 0.1 --los_C 0.5 --pheno_C 1.0 --target_repl_coef 0.5
        
##### Multitask channel-wise LSTM
        python -um mimic3models.multitask.main --network mimic3models/keras_models/multitask_channel_wise_lstms.py --dim 16 --depth 1 --batch_size 8 --timestep 1.0 --load_state mimic3models/multitask/keras_states/r2k_channel_wise_lstms.n16.szc8.0.dep1.bs8.ts1.0_partition=custom_ihm=0.1_decomp=0.1_los=0.5_pheno=1.0.epoch7.test0.958184612654.state --mode test --size_coef 8.0 --partition custom --ihm_C 0.1 --decomp_C 0.1 --los_C 0.5 --pheno_C 1.0
