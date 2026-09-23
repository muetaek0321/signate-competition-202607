# signate-competition-202607 - AI Engineering Challenge ～煩雑な社内ドライブをハックせよ～

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=astral&logoColor=white)](https://astral.sh/uv)
[![PyTorch](<https://img.shields.io/badge/PyTorch-2.11+_(cu128)-EE4C2C?logo=pytorch&logoColor=white>)](https://pytorch.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.3+-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Chroma](https://img.shields.io/badge/Chroma-Vector_DB-FF6F61)](https://www.trychroma.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-API-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Ruff](https://img.shields.io/badge/Ruff-0.15+-D7FF64?logo=ruff&logoColor=black)](https://astral.sh/ruff)

## 概要

本リポジトリは、SIGNATEコンペティション「**AI Engineering Challenge ～煩雑な社内ドライブをハックせよ～**」向けに構築された、高度なマルチモーダルRAG（Retrieval-Augmented Generation）システムです。

構造化・非構造化データ（Excel, Word, PowerPoint, PDF, CSV, ソースコード, 画像, Markdownなど）が混在する社内共有ドライブから、ユーザーの質問に対する正確な根拠ドキュメントを高速かつ高精度に検索し、LLM / VLMを活用して高精度な回答を生成・提出用ファイルに出力します。

コンペティションの参加記をブログ記事に書いていますので、良ければ併せて読んでみてください。

[https://fallpoke-tech.hatenadiary.jp/entry/2026/09/21/232649](https://fallpoke-tech.hatenadiary.jp/entry/2026/09/21/232649)

### 主な特徴

- **マルチフォーマット対応の独自ローダー群**: 表構造、スタイル・ハイライト情報、スライド構造、Jupyter Notebookセルなどを的確に抽出。
- **マルチモーダル推論**: 画像およびスライド資料をVLM（Vision-Language Model）で構造化テキスト化し、画像キャッシュ（`image_store.joblib`）を通じて回答生成時にも直接画像コンテキストを入力。
- **社内用語集を活用したメタデータ付与 & クエリ拡張**: ファイルパスと用語集からLLMがメタデータを事前抽出し、検索時もクエリを最適化。
- **ハイブリッド検索 & リランキング**: Chroma（MMRベクトル検索）× SudachiPy + BM25（形態素解析キーワード検索）× CrossEncoder（Reranker）による高精度リトリーバル。
- **厳格なハルシネーション抑制と自動リトライ**: コンテキストに基づかない推測を禁止し、情報不足時は確実に「わかりません」を出力する構造化出力制御。

---

## 主な機能

1. **ドキュメント解析 & チャンク分割 (`create_database.py`, `modules/custom_loader/`)**
   - 対応拡張子: `.csv`, `.tsv`, `.xlsx`, `.docx`, `.pptx`, `.pdf`, `.py`, `.ipynb`, `.md`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.json`, `.txt`, `.toml`
   - Officeファイル・PDFのパスワード保護検知（`modules/check_office_password.py`）
   - Excelのセル色・スタイル情報抽出（`ExcelStyleLoader`）
   - Word/PowerPointのスタイル抽出およびPDF経由の画像抽出
   - Jupyter Notebookのセル単位パース（`NotebookCellLoader`）
   - 画像ファイルのVLM解析（OCR、レイアウト、物体、表、キーワード抽出）とBase64キャッシュ管理（`ImageDocumentLoader`）
2. **社内情報に特化したファイルメタデータ作成 (`modules/create_file_info.py`)**
   - 社内用語集（`resource/社内用語集.docx`）を参照し、ファイルパスから会社名、カテゴリ、ディレクトリ種別、検索キーワードをLLMで抽出しベクトル化。
3. **クエリ最適化 (`modules/generate_query.py`)**
   - 質問文と社内用語集から、ストップワードを除去し同義語・略称を補完した検索クエリキーワード群を自動生成。
4. **対象ファイルの動的絞り込み (`modules/filepath_filter.py`)**
   - ファイル情報ベクトルDBとリランカーを用いて候補ファイルを抽出し、LLMが自社関連（株式会社データアステル）規定や質問意図に合わせて対象パスを特定。
5. **ハイブリッド検索 (`modules/response_generator.py`, `modules/bm25_search.py`)**
   - 全体ドキュメント、CSV専用、Excel専用の各Chromaコレクションに対するMMR（Maximal Marginal Relevance）検索。
   - `SudachiPy`（dict="full"）を用いた日本語形態素解析による `BM25Retriever` 検索。
6. **CrossEncoderによるリランキング**
   - `cl-nagoya/ruri-v3-reranker-310m` 等のCrossEncoderを用いて、検索されたドキュメントチャンクを質問との関連度順に再スコアリング。
7. **マルチモーダル回答生成 & リトライ制御**
   - Gemini API（`models/gemini-3.5-flash-lite` 等）または Ollama を使用。
   - テキストチャンクに加え、関連画像のBase64エンコードデータをマルチモーダル入力。
   - Pydantic（`Response` モデル）による構造化出力（`reason`, `answer`）。
   - 回答が「わかりません」となった場合の自動リトライ（次点候補ドキュメントを用いた再試行）。
8. **バッチ実行 & チェックポイント再開 (`generate_response.py`, `run_generate.bat`)**
   - 質問一覧CSVを逐次処理し、生成時間・検索元ファイル情報を記録した確認用CSV（`result_generated.csv`）と提出用CSV（`predictions.csv`）を自動保存。
   - 中断時のレジューム機能（`--resume` フラグ）。

---

## 技術スタック

| カテゴリ                 | 技術 / ライブラリ                                                                                             | 用途 / 備考                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **言語・ランタイム**     | Python 3.13+                                                                                                  | システム全体の実行基盤                                   |
| **パッケージ管理**       | uv                                                                                                            | 高速なパッケージ同期・仮想環境管理                       |
| **LLM / VLM**            | Google Gemini API (`langchain-google-genai`)<br>Ollama (`langchain-ollama`)                                   | 回答生成、画像キャプション、クエリ生成、ファイル絞り込み |
| **Embedding / Reranker** | `cl-nagoya/ruri-v3-310m`<br>`cl-nagoya/ruri-v3-reranker-310m` (`sentence-transformers`)                       | ベクトル化（Hugging Face）、CrossEncoderリランキング     |
| **ベクトルDB / 検索**    | Chroma (`langchain-chroma`)<br>BM25 (`rank-bm25`, `langchain-community`)                                      | ハイブリッドドキュメントリトリーバル                     |
| **形態素解析**           | SudachiPy, SudachiDict-full                                                                                   | BM25検索用トークナイズ                                   |
| **ドキュメント解析**     | openpyxl, python-docx, python-pptx, PyMuPDF, pypdf, msoffcrypto-tool, pywin32, nbformat, unstructured, Pillow | Office文書、PDF、ノートブック、画像の解析・OCR           |
| **深層学習基盤**         | PyTorch 2.11+ (CUDA 12.8 wheel), Transformers                                                                 | GPUを用いたEmbedding/Reranker推論                        |
| **データ操作**           | pandas, joblib                                                                                                | CSV入出力、ドキュメント・画像キャッシュ永続化            |
| **コード品質**           | Ruff                                                                                                          | Linter & Formatter                                       |
| **環境構築**             | Docker, VS Code Dev Containers                                                                                | コンテナ環境対応                                         |

---

## ディレクトリ構成

```text
script/
├── .agents/                      # エージェント用スキル・設定
├── .devcontainer/                # VS Code Dev Container 設定
│   └── devcontainer.json
├── modules/                      # システム中核モジュール
│   ├── custom_loader/            # 各ファイル形式用カスタムローダー群
│   │   ├── csv_chunk_loader.py
│   │   ├── excel_chunk_loader.py
│   │   ├── excel_style_loader.py
│   │   ├── image_file_loader.py
│   │   ├── markdown_loader.py
│   │   ├── msword_style_loader.py
│   │   ├── msword_to_pdf_image_loader.py
│   │   ├── notebook_cell_loader.py
│   │   ├── pdf_document_image_loader.py
│   │   ├── powerpoint_style_loader.py
│   │   └── powerpoint_to_pdf_image_loader.py
│   ├── bm25_search.py            # SudachiPy + BM25Retriever 検索
│   ├── check_office_password.py  # 暗号化/パスワード保護チェック
│   ├── check_win32com.py         # Windows win32com 稼働確認
│   ├── create_file_info.py       # 用語集に基づくファイルメタデータ作成
│   ├── embedding_models.py       # Embeddingモデル切替・ローダー
│   ├── filepath_filter.py        # LLMによる関連ファイルパス絞り込み
│   ├── generate_query.py         # 質問文からの検索クエリ生成
│   └── response_generator.py     # RAG統合回答生成パイプライン
├── resource/                     # リソースおよびDB保存ディレクトリ
│   ├── chroma/                   # ChromaベクトルDB & キャッシュ群
│   ├── pretrained/               # Hugging Faceキャッシュ (HF_HOME)
│   ├── temp/                     # 一時ファイル作業領域
│   └── 社内用語集.docx           # 参照用社内用語集
├── results/                      # 推論結果・提出ファイル出力先
├── share/                        # 入力データ
│   ├── 共有ドライブ/             # 検索対象ドキュメント群 (TARGET_DIR)
│   └── 質問回答/                 # 評価用・テスト用質問CSV
├── .env                          # 環境変数設定ファイル
├── .python-version               # Pythonバージョン (3.13)
├── Dockerfile                    # Ollama + uv ベースのDockerfile
├── create_database.py            # ベクトルDB作成・インデックス構築スクリプト
├── generate_response.py          # 回答生成・CSV出力スクリプト
├── pyproject.toml                # プロジェクト設定・依存関係定義
├── run_generate.bat              # バラメータ別一括実行バッチ
└── uv.lock                       # 依存関係ロックファイル
```

---

## セットアップ

### 1. 前提条件

- **Python**: `>= 3.13`
- **uv**: インストール済みであること（[Astral uv 公式ガイド](https://docs.astral.sh/uv/)）
- **GPU**: NVIDIA GPU（CUDA 12.8 推奨）
- **OS**: Windows（`pywin32` によるOffice/PDF連携を利用する場合）または Linux / Docker

### 2. リポジトリのクローンと環境構築

プロジェクトルートで以下を実行し、仮想環境の作成と依存関係のインストールを行います。

```bash
# 依存関係の同期（PyTorch cu128 も自動解決されます）
uv sync
```

### 3. 環境変数の設定

ルートディレクトリに `.env` ファイルを作成し、必要な設定を記述します。

```env
# 検索対象およびデータ保存パス
TARGET_DIR="./share/共有ドライブ"
DATASET_DIR="./resource/chroma"
TEMP_DIR="./resource/temp"

# Embedding設定 (huggingface / ollama / gemini)
EMBEDDING_MODE="huggingface"
EMBEDDING_MODEL_NAME="cl-nagoya/ruri-v3-310m"

# Reranker設定
RERANKER_MODEL_NAME="cl-nagoya/ruri-v3-reranker-310m"

# 回答生成LLM設定 (gemini / ollama)
GENERATE_MODE="gemini"
GOOGLE_API_KEY="your-google-api-key"
GEMINI_MODEL_NAME="models/gemini-3.5-flash-lite"

# Ollama使用時の設定（任意）
OLLAMA_API_KEY="your-ollama-api-key"
OLLAMA_MODEL_NAME="gemma4:31b-cloud"
```

---

## 起動方法・実行手順

パイプラインは「**1. データベース構築**」と「**2. 回答生成**」の2ステップで実行します。

### ステップ 1: データベースの構築 (`create_database.py`)

共有ドライブ内のドキュメントを読み込み、チャンク分割、VLM解析、ChromaベクトルDBの作成、BM25用キャッシュの生成を一括で行います。

```bash
uv run python create_database.py
```

- 処理が完了すると、`resource/chroma/` 配下にベクトルDBやキャッシュファイル（`all_documents.joblib`, `image_store.joblib`, `file_info_list.csv`）が保存されます。
- 途中でエラーが発生したファイルは `error_log.json` に記録され、処理は継続されます。

### ステップ 2: 回答の生成 (`generate_response.py`)

質問ファイル（`./share/質問回答/questions_test.csv`）を読み込み、検索・回答生成を実行します。

```bash
# 基本実行（採用ドキュメント数: 20, MMR lambda: 0.3, 出力先: results/n20_lm0.3）
uv run python generate_response.py -n 20 -lm 0.3 -p results/n20_lm0.3

# 中断した処理を途中から再開する場合（--resume）
uv run python generate_response.py -n 20 -lm 0.3 -p results/n20_lm0.3 --resume
```

#### コマンドライン引数

| 引数             | 短縮名 | 型    | 既定値    | 説明                                                                 |
| ---------------- | ------ | ----- | --------- | -------------------------------------------------------------------- |
| `--num-top-docs` | `-n`   | int   | `20`      | LLMへのプロンプトに含める上位ドキュメント数                          |
| `--lambda-mult`  | `-lm`  | float | `0.3`     | MMR検索における多様性制御パラメータ（0.0〜1.0）                      |
| `--output-path`  | `-p`   | str   | `results` | 結果出力先ディレクトリのパス                                         |
| `--resume`       | `-r`   | flag  | `False`   | 既存の `result_generated.csv` を読み込み、未回答の質問のみ処理を続行 |

### 一括実行バッチ (`run_generate.bat`)

Windows環境では、異なるハイパーパラメータの組み合わせを一括実行するバッチスクリプトが用意されています。

```bat
run_generate.bat
```

---

## 環境変数一覧

| 変数名                 | 使用箇所                        | 説明                                                | 既定値 / 設定例                    |
| ---------------------- | ------------------------------- | --------------------------------------------------- | ---------------------------------- |
| `TARGET_DIR`           | `create_database.py`            | インデックス対象となる共有フォルダのルートパス      | `./`（例: `./share/共有ドライブ`） |
| `DATASET_DIR`          | DB構築 / 生成スクリプト全般     | ChromaDBおよび各種キャッシュの永続化先ディレクトリ  | `./resource/chroma`                |
| `TEMP_DIR`             | ローダー / 一時処理             | ドキュメント変換等の一時ファイル作業フォルダ        | `./resource/temp`                  |
| `HF_HOME`              | スクリプト全般（コード内指定）  | Hugging Faceモデルのダウンロード保存先              | `./resource/pretrained`            |
| `CUDA_VISIBLE_DEVICES` | スクリプト全般（コード内指定）  | 使用するGPUデバイス番号                             | `"0"`                              |
| `EMBEDDING_MODE`       | `modules/embedding_models.py`   | Embeddingの方式 (`huggingface`, `ollama`, `gemini`) | `"huggingface"`                    |
| `EMBEDDING_MODEL_NAME` | `modules/embedding_models.py`   | 使用するEmbeddingモデル名                           | `cl-nagoya/ruri-v3-310m`           |
| `RERANKER_MODEL_NAME`  | `modules/response_generator.py` | リランキング用CrossEncoderモデル名                  | `cl-nagoya/ruri-v3-reranker-310m`  |
| `GENERATE_MODE`        | `modules/response_generator.py` | 回答生成用LLMの種別 (`gemini`, `ollama`)            | `"ollama"`（.env例: `"gemini"`）   |
| `GOOGLE_API_KEY`       | 各種LLM/VLMモジュール           | Google Gemini API利用キー                           | なし（必須）                       |
| `GEMINI_MODEL_NAME`    | `modules/response_generator.py` | 回答生成用Geminiモデル名                            | `models/gemini-3.5-flash-lite`     |
| `OLLAMA_API_KEY`       | `modules/response_generator.py` | Ollama Cloud利用時のAPIキー                         | なし                               |
| `OLLAMA_MODEL_NAME`    | `modules/response_generator.py` | Ollama利用時のモデル名                              | `gpt-oss:120b`                     |

---

## データ保存・成果物

| ファイル / ディレクトリ                | 形式         | 説明                                                                                                                                              |
| -------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `resource/chroma/`                     | ディレクトリ | ChromaベクトルDB（`shared_folder_all_documents`, `shared_folder_csv_documents`, `shared_folder_excel_documents`, `shared_folder_file_info_list`） |
| `resource/chroma/all_documents.joblib` | Joblib       | 全ドキュメントのLangChain `Document` オブジェクトリスト（BM25検索等で使用）                                                                       |
| `resource/chroma/image_store.joblib`   | Joblib       | 画像ファイルのBase64エンコードキャッシュ辞書（マルチモーダル推論用）                                                                              |
| `resource/chroma/file_info_list.csv`   | CSV          | 読み込み済みファイルパス、ファイル名、拡張子の一覧                                                                                                |
| `resource/chroma/error_log.json`       | JSON         | データベース作成時に読み込みに失敗したファイル名とスタックトレース                                                                                |
| `<output-path>/result_generated.csv`   | CSV          | 質問、生成回答、回答根拠（reason）、検索対象ファイル、参照ドキュメントファイルを含む詳細結果                                                      |
| `<output-path>/predictions.csv`        | CSV          | コンペティション提出用フォーマット（ヘッダーなし、`index,answer`）                                                                                |

---

## 開発・保守コマンド

コード品質とフォーマットの維持には `Ruff` を使用します。

```bash
# 静的解析（Lintチェック）
uv run ruff check .

# 自動修正
uv run ruff check --fix .

# コードフォーマット適用
uv run ruff format .
```

---

## 注意点・前提条件

1. **OSとOffice自動化（win32com）に関する注意**:
   - `modules/check_win32com.py` を経由してWord/PowerPoint等のPDF変換・画像化を行う機能は、**Windows OSかつMicrosoft Office製品が導入されている環境**でのみ動作します。
   - 非Windows環境（DockerやLinux/WSL）では自動的にフォールバックし、`Unstructured` やテキスト抽出ベースのローダーが使用されます。
2. **GPUメモリの確保**:
   - Reranker（`CrossEncoder`）および Hugging Face Embedding モデルは CUDA デバイス上で動作します。適切なVRAM（推奨8GB以上）を持つGPU環境で実行してください。
3. **APIキーとクォータ**:
   - Google Gemini API を用いた画像キャプション抽出（`ImageDocumentLoader`）やファイル情報抽出（`FileInfoCreator`）はファイル数に応じて多数のAPIリクエストを発行します。レート制限（RPM/TPM）に留意してください。
4. **データの機密性**:
   - `.env` 内のAPIキーや社内機密情報・共有ドライブデータはリポジトリにコミットしないよう `.gitignore` を確認してください。

---

## Author

- _[muetaek0321](https://github.com/muetaek0321)_
- _Gemini 3.8 Flash_
