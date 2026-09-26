"""
اپ وب موکاپ‌ساز تی‌شرت (Streamlit)
------------------------------------
این فایل رابط کاربری وب است. منطق اصلی پردازش ویدیو در فایل mockup.py قرار دارد
و این فایل فقط آن را با فایل‌های آپلودشده توسط کاربر صدا می‌زند.

نکته مهم: این فایل باید کنار mockup.py در همان ریپازیتوری باشد.
اجرا (برای تست محلی): streamlit run app_py.py
روی Streamlit Cloud: همین فایل را به عنوان "Main file path" انتخاب کنید.
"""

import os
import tempfile
import traceback

import streamlit as st

from mockup import process_video

st.set_page_config(page_title="موکاپ‌ساز ویدیویی تی‌شرت", page_icon="👕")

st.title("👕 موکاپ‌ساز ویدیویی تی‌شرت")
st.write(
    "یک ویدیو از فردی با تی‌شرت ساده و یک عکس طرح (PNG با پس‌زمینه شفاف) آپلود کنید؛ "
    "طرح با ردیابی بدن روی سینه‌ی ویدیو چسبانده می‌شود."
)

col1, col2 = st.columns(2)
with col1:
    video_file = st.file_uploader("ویدیوی ورودی", type=["mp4", "mov", "avi", "mkv"])
with col2:
    design_file = st.file_uploader("عکس طرح (PNG با پس‌زمینه شفاف)", type=["png"])

with st.expander("تنظیمات پیشرفته (اختیاری)"):
    width_scale = st.slider("عرض طرح نسبت به فاصله شانه‌ها", 0.2, 1.2, 0.62, 0.02)
    vert_offset = st.slider("فاصله از خط شانه تا بالای طرح", 0.0, 1.0, 0.35, 0.02)
    blend_strength = st.slider("شدت افکت سایه/چروک پارچه", 0.0, 1.0, 0.35, 0.05)
    smoothing = st.slider("صاف‌سازی حرکت بین فریم‌ها", 0.0, 0.95, 0.6, 0.05)
    model_variant = st.selectbox("مدل تشخیص بدن", ["lite", "full"], index=0)

run = st.button("🚀 ساخت موکاپ", type="primary", disabled=not (video_file and design_file))

if not video_file or not design_file:
    st.info("برای شروع، هم ویدیو و هم عکس طرح را آپلود کنید.")

if run and video_file and design_file:
    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, "input_video" + os.path.splitext(video_file.name)[1])
    design_path = os.path.join(tmp_dir, "design.png")
    output_path = os.path.join(tmp_dir, "output.mp4")

    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    with open(design_path, "wb") as f:
        f.write(design_file.getbuffer())

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_progress(frame_idx, total_frames):
        if total_frames > 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
        status_text.text(f"در حال پردازش فریم {frame_idx} از {total_frames}...")

    try:
        with st.spinner("در حال آماده‌سازی (ممکن است بار اول، دانلود مدل کمی طول بکشد)..."):
            process_video(
                video_path=video_path,
                design_path=design_path,
                output_path=output_path,
                width_scale=width_scale,
                vert_offset=vert_offset,
                blend_strength=blend_strength,
                smoothing=smoothing,
                model_variant=model_variant,
                progress_callback=update_progress,
            )
        status_text.text("تمام شد ✅")
        progress_bar.progress(1.0)

        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button(
                "⬇️ دانلود ویدیوی خروجی",
                data=f,
                file_name="mockup_output.mp4",
                mime="video/mp4",
            )
    except Exception as e:
        st.error(f"خطا در پردازش: {e}")
        st.code(traceback.format_exc())
