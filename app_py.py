import streamlit as st
import cv2
import numpy as np
from tempfile import NamedTemporaryFile

# تنظیمات صفحه وب‌اپلیکیشن
st.set_page_config(
    page_title="سیستم موکاپ ویدئویی لباس",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 استودیوی موکاپ ویدئویی لباس (شبیه‌سازی چروک و بافت پارچه)")
st.write("طرح PNG خود را آپلود کنید و آن را روی ویدیوی مدل قرار دهید تا با چین‌وچروک و نورپردازی پارچه ترکیب شود.")

# --- بخش سایدبار (تنظیمات و آپلود فایل) ---
st.sidebar.header("۱. آپلود فایل‌ها")
uploaded_video = st.sidebar.file_uploader("انتخاب ویدیوی مدل (MP4 / MOV)", type=["mp4", "mov", "avi"])
uploaded_logo = st.sidebar.file_uploader("انتخاب طرح PNG (لوگو یا طرح تیشرت)", type=["png"])

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

# --- بخش اصلی رابط کاربری ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("پیش‌نمایش ورودی‌ها")
    if uploaded_logo:
        st.image(uploaded_logo, caption="طرح PNG انتخابی شما", width=200)
    else:
        st.info("لطفاً یک طرح PNG بارگذاری کنید.")

with col2:
    st.subheader("خروجی ویدیو موکاپ")
    output_placeholder = st.empty()

# دکمه پردازش نهایی
if st.button("🚀 ساخت و رندر موکاپ ویدئویی"):
    if not uploaded_video or not uploaded_logo:
        st.warning("لطفاً هم ویدیو و هم طرح PNG را آپلود کنید!")
    else:
        with st.spinner("در حال پردازش فریم‌ها و شبیه‌سازی چروک‌های پارچه... لطفاً صبر کنید"):
            
            # ذخیره موقت ویدیو
            tfile = NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            
            # خواندن تصویر PNG همراه با کانال آلفا (Transparent)
            logo_bytes = np.asarray(bytearray(uploaded_logo.read()), dtype=np.uint8)
            logo = cv2.imdecode(logo_bytes, cv2.IMREAD_UNCHANGED)
            
            # مشخصات ویدیو
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            output_path = "output_mockup_final.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            progress_bar = st.progress(0)
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # محاسبه مختصات پویا روی سینه مدل بر اساس اسلایدرها
                curr_w = scale_size
                curr_h = int(scale_size * (logo.shape[0] / logo.shape[1]))
                
                x_c = int(width * (pos_x / 100.0))
                y_c = int(height * (pos_y / 100.0))
                
                # تغییر اندازه طرح PNG
                logo_resized = cv2.resize(logo, (curr_w, curr_h), interpolation=cv2.INTER_AREA)
                
                # بررسی کانال شفافیت طرح
                if logo_resized.shape[2] == 4:
                    b_l, g_l, r_l, a_l = cv2.split(logo_resized)
                    logo_rgb = cv2.merge((b_l, g_l, r_l))
                    mask = cv2.medianBlur(a_l, 3)
                    
                    h_l, w_l, _ = logo_resized.shape
                    
                    if y_c + h_l <= height and x_c + w_l <= width and y_c >= 0 and x_c >= 0:
                        roi = frame[y_c:y_c+h_l, x_c:x_c+w_l]
                        mask_float = (mask / 255.0) * opacity
                        
                        for c in range(3):
                            bg_channel = roi[:, :, c].astype(float)
                            fg_channel = logo_rgb[:, :, c].astype(float)
                            
                            if blend_mode == "Multiply (جذب سایه‌های چروک)":
                                blended = (bg_channel / 255.0) * (fg_channel / 255.0) * 255.0
                            elif blend_mode == "Overlay (برجسته روی بافت)":
                                blended = np.where(bg_channel < 128, (2 * bg_channel * fg_channel) / 255.0, 255 - (2 * (255 - bg_channel) * (255 - fg_channel) / 255.0))
                            else:
                                blended = fg_channel
                                
                            combined = (mask_float * blended) + ((1 - mask_float) * bg_channel)
                            roi[:, :, c] = np.clip(combined, 0, 255).astype(np.uint8)
                            
                        frame[y_c:y_c+h_l, x_c:x_c+w_l] = roi

                out.write(frame)
                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    
            cap.release()
            out.release()
            
            st.success("🎉 ویدیوی موکاپ شما با موفقیت ساخته شد!")
            st.video(output_path)