# STDN

A pytorch implementation for the paper: **[Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting](https://arxiv.org/abs/2502.12213)**

## Run the model in JiNan or PeMS:

first:
```
python prepareData.py
```
second:
```
python train.py
```

For the bundled PEMS-BAY HDF5 file, first convert it to the format used by
`conf/PEMSBAY_1dim_48.conf`:

```bash
python STDN/convert_pems_bay.py
python STDN/prepareData.py --config STDN/conf/PEMSBAY_1dim_48.conf
```

The converter writes `data/PEMS-BAY/PEMS-BAY.npz` with
`data.shape == (52116, 325, 2)`: speed and Unix timestamp.
The spatial graph is required for training and can be converted from the
official DCRNN graph file with:

```bash
python STDN/convert_dcrnn_adjacency.py
```

It writes the `from,to,distance` edge list expected by the STDN config as
`data/PEMS-BAY/PEMS-BAY.csv`.

For METR-LA, use its downloaded DCRNN graph in the same way:

```bash
python STDN/convert_dcrnn_adjacency.py \
  --input data/METR-LA/adj_mx.pkl --output data/METR-LA/METR-LA.csv
```

METR-LA can be converted with the same tool (its HDF5 table is `df`):

```bash
python STDN/convert_pems_bay.py \
  --input data/METR-LA/metr-la.h5 --key df \
  --output data/METR-LA/METR-LA.npz --num-of-vertices 207
```

The default setting is in conf/JiNan_1dim_12.conf
 
## Download the data from:

Google Drive: https://drive.google.com/drive/folders/1oo-eO41kbQS8aDyFWER66DdPT2k3k8_m?usp=sharing

and **[SSTBAN](https://github.com/guoshnBJTU/SSTBAN)**

# Environment
```
python 3.9.19
torch 2.3.0
numpy 1.26.3
```
or
```
python 3.10.14
torch 2.4.1
numpy 1.26.4
```
# JiNan dataset

For the Jinan dataset, we selected 406 intersection nodes in Jinan, China. At the same time, for safety reasons, we provided the relative longitude and latitude of the nodes in the 'JiNan of lalo.csv'.

# Citation

If you find our work is helpful, please cite as:

```
@inproceedings{cao2025spatiotemporal,
  title={Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting},
  author={Cao, Lingxiao and Wang, Bin and Jiang, Guiyuan and Yu, Yanwei and Dong, Junyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={11},
  pages={11463--11471},
  year={2025}
}
```
