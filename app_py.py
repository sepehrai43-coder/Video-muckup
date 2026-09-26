import streamlit as st
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
import os

st.set_page_config(
    page_title="استودیوی پیشرفته موکاپ ویدئویی (Perspective)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 موکاپ‌ساز پیشرفته ویدئویی (تطبیق زاویه و تیشرت مشکی)")
st.write("در این نسخه، طرح شما با قابلیت پرسپکتیو دقیقاً روی ۴ نقطه دلخواه روی تیشرت می‌نشیند.")

st.sidebar.header("۱. آپلود فایل‌ها")
uploaded_video = st.sidebar.file_uploader("انتخاب ویدیوی مدل (MP4 / MOV)", type=["mp4", "mov", "avi"])
uploaded_logo = st.sidebar.file_uploader("انتخاب تصویر طرح (PNG, JPG)", type=["png", "jpg", "jpeg", "webp"])

st.sidebar.header("۲. تنظیمات ۴ گوشه تیشرت (پرسپکتیو)")
st.sidebar.info("مختصات درصدی ۴ گوشه قرارگیری طرح روی تیشرت را مشخص کنید:")
p1_x = st.sidebar.slider("بالا-چپ: افقی X", 0, 100, 35)
p1_y = st.sidebar.slider("بالا-چپ: عمودی Y", 0, 100, 35)

p2_x = st.sidebar.slider("بالا-راست: افقی X", 0, 100, 65)
p2_y = st.sidebar.slider("بالا-راست: عمودی Y", 0, 100, 35)

p3_x = st.sidebar.slider("پایین-راست: افقی X", 0, 100, 65)
p3_y = st.sidebar.slider("پایین-راست: عمودی Y", 0, 100, 60)

p4_x = st.sidebar.slider("پایین-چپ: افقی X", 0, 100, 35)
p4_y = st.sidebar.slider("پایین-چپ: عمودی Y", 0, 100, 60)

st.sidebar.header("۳. تنظیمات رنگ برای تیشرت مشکی / تیره")
blend_mode = st.sidebar.selectbox(
    "حالت ترکیب رنگ روی پارچه", 
    ["Screen / روشن‌کننده (مخصوص تیشرت مشکی)", "Multiply (مخصوص تیشرت روشن/سفید)", "Normal (بدون افکت پارچه)"]
)
opacity = st.sidebar.slider("میزان شفافیت طرح", 0.1, 1.0, 0.95)

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
    st.subheader("گزارش پردازش و دانلود")
    debug_box = st.empty()
    download_placeholder = st.empty()

if st.button("🚀 شروع پردازش پرسپکتیو و رندر"):
    if not uploaded_video or not uploaded_logo:
        st.warning("لطفاً هم ویدیو و هم تصویر طرح را آپلود کنید!")
    else:
        log_messages = []
        def add_log(msg):
            log_messages.append(msg)
            debug_box.code("\n".join(log_messages))

        add_log("🔄 در حال ذخیره موقت ویدیو...")
        
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
            
            add_log(f"✅ ویدیو بارگذاری شد ({width}x{height} - {total_frames} فریم)")
            
            logo_bytes = np.asarray(bytearray(uploaded_logo.read()), dtype=np.uint8)
            logo = cv2.imdecode(logo_bytes, cv2.IMREAD_UNCHANGED)
            add_log("✅ تصویر طرح بارگذاری شد.")
            
            output_path = "output_perspective.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_path = "output_perspective.avi"
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
            add_log("🔄 در حال اعمال ترنسفورمیشن پرسپکتیو روی فریم‌ها...")
            frame_count = 5
            progress_bar = st.progress(0)
            
            # ابعاد مستطیل اصلی طرح
            h_l, w_l = logo.shape[:2]
            pts_src = np.float32([[0, 0], [w_l, 0], [w_l, h_l], [0, h_l]])
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # محاسبه ۴ نقطه مقصد بر اساس درصدهای اسلایدر
                dst_tl = [width * (p1_x / 100.0), height * (p1_y / 100.0)]
                dst_tr = [width * (p2_x / 100.0), height * (p2_y / 100.0)]
                dst_br = [width * (p3_x / 100.0), height * (p3_y / 100.0)]
                dst_bl = [width * (p4_x / 100.0), height * (p4_y / 100.0)]
                
                pts_dst = np.float32([dst_tl, dst_tr, dst_br, dst_bl])
                
                # محاسبه ماتریس پرسپکتیو برای خم کردن و منطبق کردن طرح
                matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
                warped_logo = cv2.warpPerspective(logo, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                
                # جداسازی ماسک و رنگ طرح
                if warped_logo.shape[2] == 4:
                    b_w, g_w, r_w, a_w = cv2.split(warped_logo)
                    logo_rgb = cv2.merge((b_w, g_w, r_w))
                    mask = a_w.astype(float) / 255.0
                else:
                    logo_rgb = warped_logo[:, :, :3]
                    # ایجاد ماسک هوشمند برای عکس‌های بدون شفافیت (حذف پس‌زمینه سیاه احتمالی)
                    gray_l = cv2.cvtColor(logo_rgb, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray_l, 15, 255, cv2.THRESH_BINARY)
                    mask = mask.astype(float) / 255.0

                mask = mask * opacity
                
                # اعمال ترکیب رنگ روی فریم ویدیو
                for c in range(3):
                    bg_c = frame[:, :, c].astype(float)
                    fg_c = logo_rgb[:, :, c].astype(float)
                    
                    if blend_mode.startswith("Screen"):
                        # مناسب تیشرت مشکی: رنگ‌ها را روشن و شفاف روی پس‌زمینه تیره مینماید
                        blended = 255.0 - (((255.0 - bg_c) * (255.0 - fg_c)) / 255.0)
                    elif blend_mode.startswith("Multiply"):
                        blended = (bg_c * fg_c) / 255.0
                    else:
                        blended = fg_c
                        
                    mask_3d = np.dstack([mask, mask, mask])
                    frame[:, :, c] = np.clip((mask_3d[:, :, c] * blended) + ((1.0 - mask_3d[:, :, c]) * bg_c), 0, 255).astype(np.uint8)

                out.write(frame)
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                
            cap.release()
            out.release()
            
            add_log(f"🎉 رندر پرسپکتیو با موفقیت تمام شد! ({frame_count} فریم)")
            st.success("ویدیوی موکاپ با پرسپکتیو آماده شد!")
            
            if os.path.exists(output_path):
                with open(output_path, "rb") as file_data:
                    download_placeholder.download_button(
                        label="📥 دانلود ویدیوی موکاپ پرسپکتیو",
                        data=file_data,
                        file_name="final_perspective_mockup.mp4",
                        mime="video/mp4"
                    )
                    
        except Exception as e:
            add_log(f"❌ خطا: {str(e)}")
