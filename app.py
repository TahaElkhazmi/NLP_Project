import gradio as gr
from chatbot import generate_response

# تعريف دالة التفاعل مع الشات بوت
def chat_with_bot(user_input, history=None):
    if history is None:
        history = []
    response = generate_response(user_input)
    history.append((user_input, response))
    return history, ""

# إعداد واجهة Gradio
with gr.Blocks() as app:
    gr.Markdown("""
    # 🤖 **شات بوت فقهي تجريبي**
    🚨 **تنبيه مهم:** هذا الشات بوت هو أداة دراسية تجريبية ولا يُعتبر مصدرًا رسميًا للفتاوى أو الأحكام الشرعية.
    📌 يُرجى مراجعة المصادر الموثوقة مثل **دار الإفتاء والهيئة العامة للأوقاف** للتحقق من صحة المعلومات الفقهية.
    """)

    chatbot = gr.Chatbot(label="المحادثة")
    msg = gr.Textbox(placeholder="🟢 اطرح سؤالك الفقهي هنا...", lines=1, interactive=True)
    clear_btn = gr.Button("🧹 مسح الدردشة")

    msg.submit(chat_with_bot, inputs=[msg, chatbot], outputs=[chatbot, msg])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

# تشغيل التطبيق
if __name__ == "__main__":
    app.launch(share=True)
