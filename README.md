# AMFP

## Dataset Download 
```shell
python -m AMFP.data.parse_d4rl
```
## Pretraining Phase
```shell
python -m AMFP.train env_name=$dataset_name model=amfpnet exp=gym_trl
```
## Fine-tuning Phase
```shell
python -m AMFP.train env_name=$dataset_name model=policynet exp=gym_pl model.model_config.tjn_ckpt_path=$tjn_ckpt_path
```
