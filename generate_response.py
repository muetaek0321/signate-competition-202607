import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "./resource/pretrained"  # 事前学習モデルの保存先指定

import joblib
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from sentence_transformers import CrossEncoder

from modules.bm25_search import BM25DocumentSearch
from modules.response_generator import ResponseGenerator

# 環境変数の読み込み
load_dotenv()

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    persist_directory = Path(os.getenv("DATASET_DIR", "./resource/chroma"))

    # 出力先フォルダの作成
    output_path = Path("./results")
    output_path.mkdir(exist_ok=True)

    # 質問データの読み込み
    question_df = pd.read_csv("./share/質問回答/questions_test.csv", encoding="utf-8")

    # ResponseGeneratorのインスタンス化
    response_generator = ResponseGenerator(persist_directory)

    # 途中再開か新規作成かで分岐
    if args.resume:
        answer_df = pd.read_csv(output_path / "result_generated.csv", encoding="utf-8-sig")
        answers = answer_df.to_dict(orient="list")
    else:
        answers = {"index": [], "question": [], "answer": [], "reason": [], "search_files": [], "doc_files": []}

    # 質問に対する回答の生成
    for idx, row in question_df.iterrows():
        start_time = time.perf_counter()
        input_question = row["question"]
        print(idx, input_question)

        if row["index"] in answers["index"]:
            print("  回答済みのためスキップ")
            continue
        
        # 検索の実行
        search_files = response_generator.search_context(input_question)

        # 回答の生成
        response_dict, doc_files = response_generator.genrate_answer(input_question)

        print(f"回答: {response_dict}")

        # 生成された返答内容を格納
        answers["index"].append(row["index"])
        answers["question"].append(input_question)
        answers["answer"].append(response_dict["answer"].replace("\n", ""))
        answers["reason"].append(response_dict["reason"].replace("\n", ""))
        answers["search_files"].append(",".join(search_files))
        answers["doc_files"].append(",".join(doc_files))

        generate_time = time.perf_counter() - start_time
        print(f"返答の生成時間: {generate_time:.2f}s")

        answer_df = pd.DataFrame(answers)
        # 確認用ファイルを作成
        answer_df.to_csv(
            output_path / "result_generated.csv",
            encoding="utf-8-sig",
            index=None,
        )

        # 提出用ファイルを作成
        answer_df[["index", "answer"]].to_csv(
            output_path / "predictions.csv", encoding="utf-8-sig", index=None, header=None
        )


if __name__ == "__main__":
    main()
