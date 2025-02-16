import os
import logging
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

# إعداد تسجيل المعلومات
logging.basicConfig(level=logging.INFO)

# تحميل مفتاح OpenAI API
def configure_openai_api_key():
    if os.getenv("OPENAI_API_KEY") is None:
        with open('key.txt', 'r') as f:
            os.environ["OPENAI_API_KEY"] = f.readline().strip()
    assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "Invalid OpenAI API key"
    logging.info("✅ OpenAI API key configured")

configure_openai_api_key()

# تحميل قاعدة البيانات المتجهية
def load_vector_store(vector_store_path="./vector_store"):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return Chroma(persist_directory=vector_store_path, embedding_function=embeddings)

# استرجاع المستندات ذات الصلة
def retrieve_documents(query, top_k=10):
    db = load_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)

    if not docs:
        logging.warning("⚠️ لم يتم العثور على أي مستندات ذات صلة.")
        return []

    return docs

# التأكد من أن السؤال فقهي
def is_fiqh_related(query, relevant_docs):
    keywords = ["حكم", "شروط", "الصلاة", "الزكاة", "الحج", "الصوم", "الطهارة", "الوضوء", "الكفارة", "الطلاق",
                "الميراث", "يجوز", "لا يجوز", "القصر", "الركعة", "التيمم", "الطواف", "الإحرام", "النية", "الفدية",
                "السعي", "ما حكم", "كيف", "هل يجوز", "ما هي", "ما هو", "متى", "هل يجب"]
    return any(word in query for word in keywords) or bool(relevant_docs)

# توليد الإجابة بناءً على قاعدة البيانات فقط
def generate_response(query):
    logging.info(f"🔍 Querying: {query}")
    relevant_docs = retrieve_documents(query)

    if not is_fiqh_related(query, relevant_docs):
        logging.warning("🚫 السؤال غير فقهي.")
        return "❌ يُرجى طرح سؤال فقهي فقط."

    if not relevant_docs:
        return "❌ لم أجد إجابة مباشرة لهذا السؤال، يُرجى البحث في المصادر الموثوقة."

    # **تنظيم المستندات داخل Full Prompt**
    context_sections = []
    for idx, doc in enumerate(relevant_docs, 1):
        context_sections.append(f"📄 **المصدر {idx}:**\n{doc.page_content[:1000]}...\n")

    context = "\n\n".join(context_sections)

    # **تحسين `full_prompt` لمنع التوليد مع السماح بإعادة الصياغة عند توفر المعلومات**
    full_prompt = f"""
📖 **المعلومات المسترجعة من قاعدة البيانات:**
{context}

📝 **تعليمات للنموذج:**
- **يجب أن تستخلص الإجابة فقط مما هو موجود في المعلومات المسترجعة أعلاه**.
- **يمكنك إعادة صياغة المعلومات ولكن لا تضف أي تفاصيل غير موجودة في المستندات**.
- **إذا لم تكن هناك معلومات كافية، فقط أذكر "المعلومات غير متوفرة"**.
- **لا تفسر أو تضيف أي رأي شخصي، بل استخدم المعلومات المتاحة فقط**.

🔹 **السؤال:** {query}
"""

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)  # تقليل الإبداع إلى 0
    refined_response = llm.invoke([full_prompt])

    response_text = refined_response.content.strip()

    # ✅ **منع توليد إجابة إذا لم تكن هناك معلومات كافية**
    if "المعلومات غير متوفرة" in response_text or response_text == "":
        return "❌ لم أجد إجابة مباشرة لهذا السؤال، يُرجى البحث في المصادر الرسمية."

    formatted_response = f"""
📌 **السؤال:** {query}

📖 **الإجابة:**
{response_text}

🔗 **المصدر الأول:** [اضغط هنا]({relevant_docs[0].metadata.get('url', '#')})

✅ **إذا احتجت إلى مزيد من التفاصيل، يُرجى مراجعة المصادر الموثوقة مثل دار الإفتاء والهيئة العامة للأوقاف.**
    """

    return formatted_response

if __name__ == "__main__":
    print("""
🚨 **تنبيه مهم:**
🤖 هذا الشات بوت هو أداة دراسية تجريبية ولا يُعتبر مصدرًا رسميًا للفتاوى أو الأحكام الشرعية.
📌 يُرجى مراجعة المصادر الموثوقة مثل **دار الإفتاء والهيئة العامة للأوقاف** للتحقق من صحة المعلومات الفقهية.
    """)

    while True:
        user_input = input("🟢 اطرح سؤالك الفقهي: ")
        if user_input.lower() in ["exit", "خروج"]:
            print("🔴 تم إنهاء المحادثة.")
            break
        answer = generate_response(user_input)
        print(f"🤖 الجواب: {answer}\n")
