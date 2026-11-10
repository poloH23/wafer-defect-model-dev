# data/raw

The MixedWM38 dataset is not part of this repository. Download it from
[Kaggle](https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets?select=Wafer_Map_Datasets.npz),
unzip it, and place `Wafer_Map_Datasets.npz` in this folder:
`data/raw/Wafer_Map_Datasets.npz`. Every run checks the file's hash against the committed splits
in `data/splits/` — the committed hash is for the copy from that source, so other mirrors may not
match. See [Setup › Data](../../README.md#data) for the source, its citation and its terms.

MixedWM38 資料集不包含在本 repo 中。請從 [Kaggle](https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets?select=Wafer_Map_Datasets.npz) 下載，
解壓縮後把`Wafer_Map_Datasets.npz` 放在這個資料夾，也就是 `data/raw/Wafer_Map_Datasets.npz`。
每次執行都會比對資料檔與 `data/splits/` 中固定切分的雜湊值 —— 已提交的雜湊值對應的是該來源的副本，其他鏡像未必相符。
資料來源、引用方式與使用條款見 [README](../../README.zh-TW.md#<正確的 anchor>)。
