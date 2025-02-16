import json
import csv
import os
from langchain_openai import ChatOpenAI
from tqdm import tqdm  # لإضافة Progress Bar

# تحديد أسماء ملفات JSON التي تحتوي على البيانات الفقهية
data_files = ["data1.json", "data2.json", "data3.json", "data4.json"]

# اسم ملف CSV الذي سيتم إنشاؤه
evaluation_csv = "evaluation_dataset.csv"

# ضبط مفتاح OpenAI تلقائيًا إذا لم يكن مضبوطًا
if os.getenv("OPENAI_API_KEY") is None:
    with open("key.txt", "r") as f:
        os.environ["OPENAI_API_KEY"] = f.readline().strip()
    print("✅ تم ضبط مفتاح OpenAI تلقائيًا من key.txt")


def load_json_data(file_path):
    """ تحميل البيانات من ملف JSON """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_smart_question(title, content):
    """ استخدام GPT-4o-mini لإنشاء أسئلة ذكية بناءً على العنوان والمحتوى """
    prompt = f"""
    أنت مساعد ذكي متخصص في إنشاء أسئلة فقهية للاختبار.
    العنوان: {title}
    المحتوى: {content[:500]}...

    قم بإنشاء سؤال فقهي مناسب بناءً على المعلومات المتوفرة.
    """

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    response = llm.invoke([prompt])
    return response.content.strip()


def extract_questions_and_answers():
    """ استخراج الأسئلة الذكية والإجابات المتوقعة من ملفات JSON """
    dataset = []
    total_questions = 0
    all_entries = []

    for file in data_files:
        if not os.path.exists(file):
            print(f"تحذير: الملف {file} غير موجود، سيتم تخطيه.")
            continue

        data = load_json_data(file)
        for category, entries in data.items():
            all_entries.extend(entries)

    print(f"📊 عدد الإدخالات الإجمالي: {len(all_entries)}")

    for entry in tqdm(all_entries, desc="🔄 إنشاء الأسئلة", unit=" سؤال"):
        question = generate_smart_question(entry['lecture_title'], entry['content'])
        answer = entry['content'][:500] + "..."  # اقتباس جزء من المحتوى كإجابة
        dataset.append((question, answer))
        total_questions += 1

    print(f"✅ تم إنشاء {total_questions} سؤالًا في مجموعة التقييم.")
    return dataset


def save_to_csv(dataset, file_path):
    """ حفظ مجموعة الاختبار في ملف CSV """
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Question", "Expected Answer"])
        for idx, (question, answer) in enumerate(dataset, start=1):
            writer.writerow([idx, question, answer])


def main():
    print("📌 يتم الآن إنشاء مجموعة الاختبار باستخدام GPT-4o-mini...")
    dataset = extract_questions_and_answers()
    save_to_csv(dataset, evaluation_csv)
    print(f"✅ تم إنشاء ملف التقييم: {evaluation_csv} بنجاح!")


if __name__ == "__main__":
    main()
