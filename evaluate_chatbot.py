import csv
import os
import json
import random
from tqdm import tqdm
from chatbot import generate_response  # استدعاء الشات بوت الفعلي
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util

# تحميل نموذج التشابه الدلالي
model = SentenceTransformer("all-MiniLM-L6-v2")

# اسم ملف التقييم وملف التقرير
evaluation_csv = "evaluation_dataset.csv"
evaluation_report = "evaluation_summary.txt"


def load_evaluation_dataset(sample_size=150):
    """ تحميل بيانات التقييم من ملف CSV واختيار عينة عشوائية """
    dataset = []
    with open(evaluation_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append((row["Question"], row["Expected Answer"]))
    return random.sample(dataset, min(sample_size, len(dataset)))  # اختيار عينة عشوائية


def evaluate_chatbot():
    """ تنفيذ عملية التقييم ومقارنة الإجابات """
    dataset = load_evaluation_dataset()
    total_questions = len(dataset)
    correct_answers = 0
    partial_correct_answers = 0
    incorrect_responses = []

    print(f"📌 بدء التقييم لـ {total_questions} سؤالًا...")

    for question, expected_answer in tqdm(dataset, desc="🔍 تقييم الإجابات", unit=" سؤال"):
        actual_answer = generate_response(question)  # استدعاء الشات بوت الفعلي

        # حساب التشابه باستخدام Fuzzy Matching
        fuzzy_similarity = fuzz.ratio(expected_answer.strip().lower(), actual_answer.strip().lower())

        # حساب التشابه باستخدام Embeddings
        embeddings_similarity = util.pytorch_cos_sim(
            model.encode(expected_answer, convert_to_tensor=True),
            model.encode(actual_answer, convert_to_tensor=True)
        ).item() * 100

        # تحديد مدى صحة الإجابة
        if embeddings_similarity >= 80 or fuzzy_similarity >= 85:
            correct_answers += 1
        elif embeddings_similarity >= 60 or fuzzy_similarity >= 70:
            partial_correct_answers += 1
        else:
            incorrect_responses.append(
                (question, expected_answer, actual_answer, fuzzy_similarity, embeddings_similarity))

    accuracy = (correct_answers / total_questions) * 100
    partial_accuracy = (partial_correct_answers / total_questions) * 100
    error_rate = 100 - (accuracy + partial_accuracy)

    with open(evaluation_report, 'w', encoding='utf-8') as f:
        f.write(f"✅ تقرير شامل عن أداء الشات بوت\n")
        f.write(f"📊 إجمالي الأسئلة التي تم تقييمها: {total_questions}\n")
        f.write(f"📊 نسبة الإجابات الصحيحة: {accuracy:.2f}%\n")
        f.write(f"📊 نسبة الإجابات الجزئية الصحيحة: {partial_accuracy:.2f}%\n")
        f.write(f"📊 نسبة الأخطاء: {error_rate:.2f}%\n\n")

        f.write("❌ أمثلة على الإجابات غير الصحيحة:\n")
        for question, expected, actual, fuzzy_sim, emb_sim in incorrect_responses[:10]:  # عرض 10 أمثلة فقط
            f.write(f"🔹 السؤال: {question}\n")
            f.write(f"✅ الإجابة الصحيحة: {expected}\n")
            f.write(f"❌ إجابة الشات بوت: {actual}\n")
            f.write(f"🔢 نسبة التشابه (Fuzzy): {fuzzy_sim:.2f}% | (Embeddings): {emb_sim:.2f}%\n\n")

    print(
        f"✅ التقييم مكتمل!\n📊 نسبة الإجابات الصحيحة: {accuracy:.2f}%\n📊 نسبة الإجابات الجزئية: {partial_accuracy:.2f}%\n📊 نسبة الأخطاء: {error_rate:.2f}%\n")
    print(f"📄 تم حفظ التقرير في {evaluation_report}")


def main():
    evaluate_chatbot()


if __name__ == "__main__":
    main()
