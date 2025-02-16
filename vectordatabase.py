import argparse
import json
import logging
import os
import pathlib
from typing import List, Tuple

import langchain
import wandb
from langchain_community.cache import SQLiteCache
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


# ✅ إعداد مفتاح OpenAI API من `key.txt` أو المتغيرات البيئية
def configure_openai_api_key():
    """قراءة مفتاح OpenAI API من `key.txt` وتخزينه كمتغير بيئي"""
    key_file = "key.txt"

    if os.getenv("OPENAI_API_KEY") is None and os.path.exists(key_file):
        with open(key_file, 'r', encoding="utf-8") as f:
            os.environ["OPENAI_API_KEY"] = f.readline().strip()

    assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "❌ خطأ: مفتاح OpenAI API غير صالح أو مفقود!"
    print("✅ OpenAI API key تم تكوينه بنجاح")


configure_openai_api_key()

langchain.llm_cache = SQLiteCache(database_path="langchain.db")
logger = logging.getLogger(__name__)


# ✅ تحميل المحاضرات من ملف JSON
def load_documents(json_file: str) -> List[Document]:
    """تحميل البيانات من JSON وتحويلها إلى كائنات Document في LangChain"""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for category, lectures in data.items():
        for lecture in lectures:
            content = lecture.get("content", "")
            metadata = {
                "lecture_title": lecture.get("lecture_title", "Unknown Title"),
                "lecture_url": lecture.get("lecture_url", "Unknown URL"),
                "category": category,
            }
            documents.append(Document(page_content=content, metadata=metadata))
    return documents


# ✅ تقسيم المستندات إلى أجزاء أصغر لتحسين البحث
def chunk_documents(documents: List[Document], chunk_size: int = 700, chunk_overlap=150) -> List[Document]:
    """تقسيم المستندات إلى أجزاء أصغر لتحسين الفهرسة والبحث"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


# ✅ إنشاء قاعدة بيانات المتجهات في Chroma
def create_vector_store(documents, vector_store_path: str = "./vector_store") -> Chroma:
    """إنشاء قاعدة بيانات Chroma وتخزين البيانات فيها"""
    api_key = os.environ.get("OPENAI_API_KEY", None)
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=vector_store_path,
    )

    print("✅ تم حفظ قاعدة البيانات تلقائيًا في ChromaDB")
    return vector_store


# ✅ تسجيل البيانات في Weights & Biases (wandb)
def log_dataset(documents: List[Document], run: "wandb.run"):
    """تسجيل بيانات المستندات كـ Dataset في wandb"""
    document_artifact = wandb.Artifact(name="dorar_dataset", type="dataset")
    with document_artifact.new_file("documents.json", mode="w", encoding="utf-8") as f:
        for document in documents:
            f.write(document.model_dump_json() + "\n")  # ✅ إصلاح الترميز + التوافق مع Pydantic V2
    run.log_artifact(document_artifact)


def log_index(vector_store_dir: str, run: "wandb.run"):
    """تسجيل قاعدة بيانات المتجهات في wandb"""
    index_artifact = wandb.Artifact(name="vector_store", type="search_index")
    index_artifact.add_dir(vector_store_dir)
    run.log_artifact(index_artifact)


def log_prompt(prompt: dict, run: "wandb.run"):
    """تسجيل الـ Prompt المستخدم في wandb"""
    prompt_artifact = wandb.Artifact(name="chat_prompt", type="prompt")
    with prompt_artifact.new_file("prompt.json", mode="w", encoding="utf-8") as f:
        f.write(json.dumps(prompt))
    run.log_artifact(prompt_artifact)


# ✅ تنفيذ عملية الإدخال والتخزين
def ingest_data(json_file: str, chunk_size: int, chunk_overlap: int, vector_store_path: str) -> Tuple[
    List[Document], Chroma]:
    """إدخال بيانات المحاضرات الإسلامية من JSON إلى قاعدة بيانات المتجهات"""
    documents = load_documents(json_file)
    split_documents = chunk_documents(documents, chunk_size, chunk_overlap)
    vector_store = create_vector_store(split_documents, vector_store_path)
    return split_documents, vector_store


# ✅ تحليل المدخلات الخاصة بالبرنامج
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="ملف JSON الذي يحتوي على بيانات المحاضرات",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="حجم الأجزاء لكل وثيقة",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="عدد الكلمات المتداخلة بين الأجزاء",
    )
    parser.add_argument(
        "--vector_store",
        type=str,
        default="./vector_store",
        help="المجلد الذي سيتم حفظ قاعدة بيانات المتجهات فيه",
    )
    parser.add_argument(
        "--prompt_file",
        type=pathlib.Path,
        default="./chat_prompt.json",
        help="ملف JSON يحتوي على الحوافز (Prompts)",
    )
    parser.add_argument(
        "--wandb_project",
        default="dorar_project",
        type=str,
        help="اسم مشروع wandb لتخزين البيانات",
    )
    return parser


# ✅ تشغيل البرنامج
def main():
    parser = get_parser()
    args = parser.parse_args()

    # ✅ تشغيل wandb لتتبع البيانات
    run = wandb.init(project=args.wandb_project, config=args, mode="offline")  # تشغيله بدون رفع البيانات

    # ✅ تنفيذ عملية الإدخال والتخزين
    documents, vector_store = ingest_data(
        json_file=args.json_file,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_store_path=args.vector_store,
    )

    # ✅ تسجيل البيانات في wandb
    log_dataset(documents, run)
    log_index(args.vector_store, run)
    log_prompt(json.load(args.prompt_file.open("r")), run)

    run.finish()


if __name__ == "__main__":
    main()
