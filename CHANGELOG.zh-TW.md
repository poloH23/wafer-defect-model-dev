# 變更紀錄

本檔案記錄這個專案所有值得注意的變更。

格式遵循 [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)，版號遵循
[語意化版本](https://semver.org/lang/zh-TW/)。

記錄範圍：本檔案記錄的是「**使用**這個 repo 的人會感受到的變化」。實驗 run 與其指標
記錄在 MLflow，建模決策的理由記錄在 `docs/`。

English: [CHANGELOG.md](CHANGELOG.md)

## [未發布]

### 新增

- `scripts/plot_showcase_figure.py` 與 `make showcase-figure`：涵蓋全部十個候選架構的
  準確度對延遲圖，中英文各一份，數據取自已提交的報告表格。

### 修正

- 資料集取得說明原先指向作者的 repo，該 repo 有 MixedWM38 的說明文件但並未提供
  `Wafer_Map_Datasets.npz`。現已改指向 Kaggle —— 其 SHA-256 與已提交的切分摘要值相符 ——
  並補上資料集作者提供的引用格式與使用條款。

## [1.0.2] - 2026-11-08

處理 1.0.0 所建立的排程 dependency audit 首次回報的問題，並補上這次處理過程本身所
暴露出來的防護機制。

### 資安

- 將 `hydra-core` 的下限提高至 `>= 1.3.8`，修補 GHSA-wf6j-7hvr-gr93（CVSS 3.1
  7.8，High）。Hydra 的 legacy target blocklist 未能阻擋 PyYAML 的 unsafe loading
  函式，若不可信的 config 進入 legacy `instantiate()` / `call()` 路徑，可導致任意
  程式碼執行。本專案僅載入自有 config，實際暴露風險低。
- 將 `pyasn1` 的下限提高至 `>= 0.6.5`，修補 CVE-2026-119194、-119195、-119196、
  -119197（各 CVSS 3.1 7.5，High）。BER/CER/DER 解碼器存在 quadratic-complexity
  阻斷服務漏洞。經 `google-auth` 間接引入；訓練流程不解碼外部提供的 ASN.1 輸入。
- 兩者皆以下限形式記錄於 `constraints.txt`，而非交由 resolver 決定，因此後續重新
  編譯不會無聲地退回有漏洞的版本。

### 新增

- `tools/check_requirements.py`：一次列出 requirements 檔案中**所有**無法解析的行，
  並檢查同時出現在多個 lock 檔的套件版本是否一致。`pip-audit` 與 `pip` 都只會停在
  第一個壞行，否則每一個壞行都要耗掉一次 CI 執行。
- `make check-env`：當前作用中的 conda 環境與 `ENV` 不符時直接失敗。

### 變更

- **`make requirements`、`test`、`sanity-check`、`train`、`export-onnx` 現在會拒絕
  在專案 conda 環境之外執行。** `conda-dev` 不受限（它的工作就是建立該環境），
  `audit` 不受限（它只讀 lock 檔）。
- `pip` 由 26.2.2（vendored certifi 2026.10.19）釘選至 26.3（vendored certifi
  2026.10.23、truststore 0.10.5）。並無對應的 advisory；此次提升是為了讓 vendored
  信任庫保持在最新狀態 — 那正是這個釘選存在的理由。
- CI：`actions/checkout` v4 → v5、`actions/setup-python` v5 → v6。兩者原先仍宣告
  `node20`，而 GitHub 已於 2026-09-23 自 Actions runner 移除 Node 20，導致它們被
  強制以 Node 24 執行並產生 deprecation 警告。v5 與 v6 分別是兩個 action 首個改用
  Node 24 的版本。
- CI：`runs-on: ubuntu-latest` → `ubuntu-26.04`。GitHub 自 2026-10-19 起將
  `ubuntu-latest` 標籤遷移至 Ubuntu 26.04（預計 2026-11-19 完成），理由是該版本已達
  production 可用的 GA 狀態 — 屬例行的 image 生命週期更新，並非資安事件因應。主動
  釘選可維持 CI 可重現性，並避免標籤語意在遷移期間改變。
- 覆蓋率統計排除 `tests/` 與 `tools/`，它們屬於開發工具而非產品程式碼。

## [1.0.1] - 2026-10-26

### 新增

- 針對乾淨 clone 的環境建置指南。

### 變更

- 統一為單一條 conda 建置路徑，本機建置與 CI 共用，因此 `make conda-dev` 與 CI job
  執行的是同一份腳本。`src` 的安裝納入環境初始化流程，因此乾淨 clone 後即可建立環境
  並執行每一個步驟。
- 兩份 README 於公開前完成並校正。
- 整理 repository 設定。

## [1.0.0] - 2026-10-24

初始版本：完整的模型開發專案，從資料集分析到交付 ONNX 模型，使用 Mixed-type Wafer
Map Defect 資料集。

### 新增

- **資料集 EDA**：NPZ 分析、影像與 CSV 封裝、通用影像讀取器，以及對照官方文件標示
  範圍的像素值分布報告，並提供未記載像素值的可設定處理政策。
- **資料流程**：config 驅動的固定 train/val/test 切分，支援排除特定樣本與移除完全
  重複的影像（split v2）。檔案雜湊在切分與訓練之間共用，因此每次 run 都會記錄其
  資料來源譜系。
- **設定系統**：模型與訓練的具型別 schema，並對照類別數驗證。
- **模型**：涵蓋具小輸入 stem 的 ResNet、MobileNetV3、EfficientNet-B0、
  ShuffleNetV2 的工廠，各自採用一致的小輸入調整，另加一系列 slim ResNet。所有模型
  都在同一處依其 config 建構。
- **訓練**：多標籤 BCE 搭配計算得出的 `pos_weight`、免閾值的 macro AP、early
  stopping、learning-rate warmup、final-epoch 協定、單批次 sanity check，以及錨定
  專案根目錄的 MLflow 追蹤。seed 與 learning rate 可逐次覆寫。
- **超參數搜尋**：以 Optuna 對 slim ResNet 進行可續跑的多目標搜尋（驗證誤差對
  FLOPs），並可將選定的 trial 匯出為確認實驗用的 config。
- **效能量測**：針對三個佈署目標（cloud GPU、CPU server、edge）量測模型大小、延遲
  與吞吐量。
- **評估**：自 MLflow run 以 seed 為單位比較實驗 config；驗證誤差分析，包含逐樣本
  機率與錯誤信心度，並附掛回該 run；基於規則的 Edge_Loc / Edge_Ring 邊緣覆蓋診斷；
  閾值調校；bootstrap 信賴區間；以及拒絕重複執行的單次 test split 評估。
- **可解釋性**：Grad-CAM 分析，含定位、方向與隨機化檢查，另加 Edge_Loc 的邊緣輪廓
  診斷。
- **交付**：將選定模型匯出、驗證並註冊為自帶權重的 ONNX 檔案，並產生 model card。
  ShuffleNetV2 1.0x 在三個佈署目標上皆為最佳選擇；選型協定與結果見
  `docs/model_selection.md`。
- **相依管理**：手工維護的 `.in` 檔案以 pip-compile 編譯為兩份 lock 檔，其中
  `requirements.txt` 受 `requirements-dev.txt` 約束，因此兩者釘選相同版本。
  `make requirements` 重新編譯；`UPGRADE="pkg ..."` 可釋放指定的釘選。
- **文件**：模型選型協定、搜尋設計、方法論、結果與限制 — 包含 Edge_Loc /
  Edge_Ring 標籤歧義 — 每份文件皆附繁體中文版本。
- Apache 2.0 授權。

### 資安

- CI 中的相依套件稽核：以 `pip-audit` 檢查 lock 檔，每週執行並於每次 push 觸發。
- `pip` 改由帶 hash 的直接 URL 安裝，因此即使 TLS 連線遭攔截，bootstrap 仍可驗證。
  此措施是為因應 eMudhra / emSign CA 憑證事件 — 舊版 pip 會透過其 vendored certifi
  信任該 CA，與系統信任庫的設定無關。
- 針對開發期間由稽核發現的已知漏洞套件，在 `constraints.txt` 記錄安全下限，包含
  `certifi` 與 `virtualenv`，以及在本版本審查期間才公布 advisory 的 `filelock` 與
  `mlflow`。

[未發布]: https://github.com/poloH23/wafer-defect-model-dev/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/poloH23/wafer-defect-model-dev/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/poloH23/wafer-defect-model-dev/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/poloH23/wafer-defect-model-dev/releases/tag/v1.0.0
