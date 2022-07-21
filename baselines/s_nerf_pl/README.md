# Spherical-NeRF
This implementation adapts the NeRF code from [**This Project**](https://kwea123.github.io/nerf_pl/).

## Datasets + Models
For data and pre-trained models please visit the [**ReadMe**](README.md). .


### Training
```
cd baselines/s_nerf_pl

# Replica
bash scripts/replica.sh # Scene number goes from 00 to 11
# Residential Area
bash scripts/residential.sh # Scene number goes from 0 to 2
```

### Evaluations
```
# Check the following scripts
bash scripts/replica_test.sh
bash scripts/residential_test.sh
```

