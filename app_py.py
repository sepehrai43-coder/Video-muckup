import streamlit as st
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
import os

st.set_page_config(
    page_title="سیستم موکاپ ویدئویی پیشرفته",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 استودیوی موکاپ ویدئویی لباس")
st.write("ویدیوی مدل و تصویر طرح خود را آپلود کنید تا با چین‌وچروک و نورپردازی پارچه ترکیب شود.")

st.sidebar.header("۱. آپلود فایل‌ها")
uploaded_video = st.sidebar.file_uploader("انتخاب ویدیوی مدل (MP4 / MOV)", type=["mp4", "mov", "avi"])
uploaded_logo = st.sidebar.file_uploader("انتخاب تصویر طرح (PNG, JPG, JPEG, WEBP)", type=["png", "jpg", "jpeg", "webp"])

st.sidebar.header("۲. تنظیمات ابعاد و جایگذاری")
pos_x = st.sidebar.slider("موقعیت افقی (X)", 0, 100, 35)
pos_y = st.sidebar.slider("موقعیت عمودی (Y)", 0, 100, 40)
scale_size = st.sidebar.slider("اندازه طرح (Scale)", 50, 500, 200)

st.sidebar.header("۳. تنظیمات واقع‌گرایانه (چروک و سایه)")
blend_mode = st.sidebar.selectbox(
    "حالت ترکیب رنگ و سایه پارچه", 
    ["Multiply (جذب سایه‌های چروک)", "Overlay (برجسته روی بافت)", "Normal (معمولی)"]
)
opacity = st.sidebar.slider("میزان شفافیت/محو شدن در چروک‌ها", 0.1, 1.0, 0.85)

col1, col2 = st.columns(2)

with col1:
    st.subheader("پیش‌نمایش ورودی‌ها")
    if uploaded_video:
        st.write("🎥 **پیش‌نمایش ویدیو:**")
        st.video(uploaded_video)
    else:
        st.info("لطفاً یک ویدیو بارگذاری کنید.")
        
    if uploaded_logo:
        st.write("🖼️ **پیش‌نمایش طرح:**")
        st.image(uploaded_logo, width=200)

with col2:
    st.subheader("گزارش پردازش و دانلود خروجی")
    debug_box = st.empty()
    download_placeholder = st.empty()

if st.button("🚀 شروع پردازش و رندر موکاپ"):
    if not uploaded_video or not uploaded_logo:
        st.warning("لطفاً هم ویدیو و هم تصویر طرح را آپلود کنید!")
    else:
        log_messages = []
        def add_log(msg):
            log_messages.append(msg)
            debug_box.code("\n".join(log_messages))

        add_log("🔄 در حال ذخیره موقت فایل ویدیو...")
        
        try:
            tfile = NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            add_log(f"✅ ویدیو بارگذاری شد ({total_frames} فریم)")
            
            logo_bytes = np.asarray(bytearray(uploaded_logo.read()), dtype=np.uint8)
            logo = cv2.imdecode(logo_bytes, cv2.IMREAD_UNCHANGED)
            add_log("✅ تصویر طرح بارگذاری شد.")
            
            output_path = "output_mockup.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_path = "output_mockup.avi"
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
            add_log("🔄 در حال اعمال طرح روی فریم‌های ویدیو...")
            frame_count = 0
            progress_bar = st.progress(0)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                curr_w = scale_size
                curr_h = int(scale_size * (logo.shape[0] / logo.shape[1]))
                
                x_c = int(width * (pos_x / 100.0))
                y_c = int(height * (pos_y / 100.0))
                
                logo_resized = cv2.resize(logo, (curr_w, curr_h), interpolation=cv2.INTER_AREA)
                
                if len(logo_resized.shape) == 3 and logo_resized.shape[2] == 4:
                    b_l, g_l, r_l, a_l = cv2.split(logo_resized)
                    logo_rgb = cv2.merge((b_l, g_l, r_l))
                    mask = cv2.medianBlur(a_l, 3)
                else:
                    logo_rgb = logo_resized[:, :, :3] if logo_resized.shape[2] >= 3 else cv2.cvtColor(logo_resized, cv2.COLOR_GRAY2BGR)
                    mask = np.full((logo_resized.shape[0], logo_resized.shape[1]), 255, dtype=np.uint8)
                
                h_l, w_l, _ = logo_resized.shape
                
                if y_c + h_l <= height and x_c + w_l <= width and y_c >= 0 and x_c >= 0:
                    roi = frame[y_c:y_c+h_l, x_c:x_c+w_l]
                    mask_float = (mask / 255.0) * opacity
                    
                    for c in range(3):
                        bg_channel = roi[:, :, c].astype(float)
                        fg_channel = logo_rgb[:, :, c].astype(float)
                        
                        if blend_mode.startswith("Multiply"):
                            blended = (bg_channel / 255.0) * (fg_channel / 255.0) * 255.0
                        elif blend_mode.startswith("Overlay"):
                            blended = np.where(bg_channel < 128, (2 * bg_channel * fg_channel) / 255.0, 255 - (2 * (255 - bg_channel) * (255 - fg_channel) / 255.0))
                        else:
                            blended = fg_channel
                            
                        combined = (mask_file_val := (mask_float * blended) + ((1 - mask_float) * bg_channel))
                        roi[:, :, c] = np.clip(combined, 0, 255).astype(np.uint8)
                        
                    frame[y_c:y_c+h_l, x_c:x_c+w_l] = roi

                out.write(frame)
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                
            cap.release()
            out.release()
            
            add_log(f"🎉 رندر با موفقیت به پایان رسید! ({frame_count} فریم پردازش شد)")
            st.success("ویدیوی موکاپ آماده شد!")
            
            # قرار دادن دکمه دانلود مستقیم برای کاربر
            if os.path.exists(output_path):
                with open(output_path, "rb") as file_data:
                    download_placeholder.download_button(
                        label="📥 دانلود ویدیوی موکاپ نهایی",
                        data=file_data,
                        file_name="final_mockup_video.mp4",
                        mime="video/mp4"
                    )
                    
        except Exception as e:
            add_log(f"❌ خطا: {str(e)}")
