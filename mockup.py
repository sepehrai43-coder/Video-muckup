#!/usr/bin/env python3
"""
مموکاپ‌ساز ویدیویی تی‌شرت
---------------------------
یک ویدیو از فردی که تی‌شرت ساده پوشیده + یک عکس PNG (با پس‌زمینه شفاف) می‌گیرد
و طرح را با ردیابی نقاط بدن (شانه‌ها) روی تی‌شرت در طول ویدیو می‌چسباند.

نحوه اجرا:
    python3 mockup.py --video input.mp4 --design design.png --output output.mp4

پارامترهای تنظیمی مهم (برای تنظیم دقیق‌تر محل و اندازه طرح):
    --width-scale      عرض طرح نسبت به فاصله دو شانه (پیش‌فرض 0.62)
    --height-scale     نسبت ارتفاع به عرض طرح (پیش‌فرض بر اساس خود PNG محاسبه می‌شود)
    --vert-offset      فاصله عمودی از خط شانه تا بالای طرح، به نسبت فاصله شانه‌ها (پیش‌فرض 0.35)
    --blend            شدت افکت چروک/سایه‌ی پارچه روی طرح، بین 0 (خاموش) تا 1 (پیش‌فرض 0.35)
"""

import argparse
import subprocess
import sys
import os
import urllib.request

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python import BaseOptions
except ImportError:
    print("خطا: کتابخانه mediapipe نصب نیست. با دستور زیر نصب کنید:")
    print("    pip install mediapipe opencv-python numpy --break-system-packages")
    sys.exit(1)


# آدرس رسمی گوگل برای مدل تشخیص پوز (نسخه lite: سریع‌تر، نسخه full: دقیق‌تر)
POSE_MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}


def ensure_pose_model(model_variant="lite"):
    """اگر فایل مدل تشخیص پوز موجود نباشد، آن را یک‌بار دانلود می‌کند."""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"pose_landmarker_{model_variant}.task")

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return model_path

    url = POSE_MODEL_URLS[model_variant]
    print(f"در حال دانلود مدل تشخیص بدن (یک‌بار، حدود چند مگابایت)...\n  {url}", file=sys.stderr)
    try:
        urllib.request.urlretrieve(url, model_path)
    except Exception as e:
        raise RuntimeError(
            "دانلود مدل تشخیص بدن ناموفق بود. اتصال اینترنت را بررسی کنید یا فایل را "
            f"دستی از این آدرس دانلود کرده و در مسیر زیر قرار دهید:\n  URL: {url}\n  مسیر: {model_path}\n"
            f"خطای اصلی: {e}"
        )
    return model_path


def load_design(path):
    """PNG طرح را با کانال آلفا بارگذاری می‌کند."""
    design = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if design is None:
        raise FileNotFoundError(f"فایل طرح پیدا نشد: {path}")
    if design.shape[2] == 3:
        # اگر آلفا نداشت، یکی می‌سازیم (کاملاً غیرشفاف)
        alpha = np.full(design.shape[:2], 255, dtype=np.uint8)
        design = cv2.merge([design[:, :, 0], design[:, :, 1], design[:, :, 2], alpha])
    return design


def get_shoulder_points(landmarks, frame_w, frame_h):
    """
    مختصات شانه چپ/راست و یک نقطه پایین‌تر (وسط سینه) را
    از نقاط پوز mediapipe استخراج می‌کند.
    landmarks: لیست NormalizedLandmark (خروجی PoseLandmarker جدید)
    """
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * frame_w, lm.y * frame_h], dtype=np.float32)

    l_sh = pt(LEFT_SHOULDER)
    r_sh = pt(RIGHT_SHOULDER)
    l_hip = pt(LEFT_HIP)
    r_hip = pt(RIGHT_HIP)

    return l_sh, r_sh, l_hip, r_hip


def build_target_quad(l_sh, r_sh, l_hip, r_hip, width_scale, height_scale, vert_offset):
    """
    بر اساس نقاط شانه و لگن، چهارضلعی محل قرارگیری طرح روی سینه را می‌سازد.
    ترتیب نقاط خروجی: بالا-چپ، بالا-راست، پایین-راست، پایین-چپ
    (مطابق با ترتیب گوشه‌های تصویر طرح در warpPerspective)
    """
    shoulder_mid = (l_sh + r_sh) / 2.0
    shoulder_width = np.linalg.norm(r_sh - l_sh)

    # بردار عمود بر خط شانه‌ها، رو به پایین (به سمت لگن) برای تعیین جهت "پایین تنه"
    torso_vec = ((l_hip + r_hip) / 2.0) - shoulder_mid
    torso_len = np.linalg.norm(torso_vec)
    if torso_len < 1e-3:
        torso_dir = np.array([0.0, 1.0], dtype=np.float32)
    else:
        torso_dir = torso_vec / torso_len

    # بردار افقی طرح، موازی خط شانه‌ها
    horiz_dir = (r_sh - l_sh)
    if shoulder_width > 1e-3:
        horiz_dir = horiz_dir / shoulder_width
    else:
        horiz_dir = np.array([1.0, 0.0], dtype=np.float32)

    design_width = shoulder_width * width_scale
    design_height = design_width * height_scale

    top_center = shoulder_mid + torso_dir * (shoulder_width * vert_offset)

    top_left = top_center - horiz_dir * (design_width / 2.0)
    top_right = top_center + horiz_dir * (design_width / 2.0)
    bottom_left = top_left + torso_dir * design_height
    bottom_right = top_right + torso_dir * design_height

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def warp_and_blend(frame, design_bgra, dst_quad, blend_strength):
    """طرح را با پرسپکتیو روی فریم می‌چسباند و برای طبیعی‌تر شدن با نسخه خاکستری فریم ترکیب می‌کند."""
    h, w = frame.shape[:2]
    dh, dw = design_bgra.shape[:2]

    src_quad = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)

    warped_design = cv2.warpPerspective(
        design_bgra, matrix, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
    )

    warped_bgr = warped_design[:, :, :3].astype(np.float32)
    warped_alpha = (warped_design[:, :, 3:4].astype(np.float32)) / 255.0

    if blend_strength > 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # نرمال‌سازی روشنایی محلی حول میانگین 0.5 تا رنگ کلی طرح خیلی تیره/روشن نشود
        gray_norm = np.clip(gray + 0.5 - float(np.mean(gray)), 0.0, 1.0)
        gray_3ch = cv2.merge([gray_norm, gray_norm, gray_norm])
        shaded = warped_bgr * gray_3ch
        warped_bgr = warped_bgr * (1 - blend_strength) + shaded * blend_strength

    frame_f = frame.astype(np.float32)
    out = frame_f * (1 - warped_alpha) + warped_bgr * warped_alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def process_video(video_path, design_path, output_path,
                   width_scale=0.62, height_scale=None, vert_offset=0.35,
                   blend_strength=0.35, smoothing=0.6, model_variant="lite",
                   progress_callback=None):
    design = load_design(design_path)
    dh, dw = design.shape[:2]
    if height_scale is None:
        height_scale = dh / dw  # نسبت طبیعی خود عکس حفظ شود

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"فایل ویدیو پیدا نشد یا باز نشد: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_video = output_path + ".temp_noaudio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (frame_w, frame_h))

    model_path = ensure_pose_model(model_variant)
    base_options = BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    smoothed_quad = None
    frame_idx = 0
    detected_count = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # timestamp به میلی‌ثانیه، باید صعودی باشد
            timestamp_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_idx += 1

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]  # اولین/تنها فرد تشخیص داده‌شده
                l_sh, r_sh, l_hip, r_hip = get_shoulder_points(landmarks, frame_w, frame_h)
                target_quad = build_target_quad(
                    l_sh, r_sh, l_hip, r_hip, width_scale, height_scale, vert_offset
                )

                if smoothed_quad is None:
                    smoothed_quad = target_quad
                else:
                    smoothed_quad = smoothing * smoothed_quad + (1 - smoothing) * target_quad

                frame = warp_and_blend(frame, design, smoothed_quad, blend_strength)
                detected_count += 1

            writer.write(frame)

            if progress_callback:
                progress_callback(frame_idx, total_frames)
            elif frame_idx % 30 == 0 or frame_idx == total_frames:
                print(f"  پردازش فریم {frame_idx}/{total_frames}...", file=sys.stderr)

    cap.release()
    writer.release()

    if detected_count == 0:
        print("هشدار: در هیچ فریمی بدن/شانه تشخیص داده نشد. خروجی بدون طرح خواهد بود.", file=sys.stderr)

    # اضافه کردن دوباره صدای ویدیوی اصلی با ffmpeg
    merge_audio(video_path, temp_video, output_path)
    os.remove(temp_video)


def merge_audio(original_video, video_no_audio, output_path):
    """صدای ویدیوی اصلی را با ffmpeg روی ویدیوی پردازش‌شده سوار می‌کند."""
    has_ffmpeg = subprocess.run(
        ["which", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).returncode == 0

    if not has_ffmpeg:
        print("هشدار: ffmpeg پیدا نشد؛ خروجی بدون صدا ذخیره می‌شود.", file=sys.stderr)
        os.replace(video_no_audio, output_path)
        return

    cmd = [
        "ffmpeg", "-y",
        "-i", video_no_audio,
        "-i", original_video,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("هشدار: ادغام صدا شکست خورد؛ خروجی بدون صدا ذخیره می‌شود.", file=sys.stderr)
        print(result.stderr.decode(errors="ignore")[-500:], file=sys.stderr)
        os.replace(video_no_audio, output_path)


def main():
    parser = argparse.ArgumentParser(description="موکاپ‌ساز ویدیویی تی‌شرت با ردیابی بدن")
    parser.add_argument("--video", required=True, help="مسیر ویدیوی ورودی")
    parser.add_argument("--design", required=True, help="مسیر عکس طرح (PNG با پس‌زمینه شفاف)")
    parser.add_argument("--output", required=True, help="مسیر ویدیوی خروجی")
    parser.add_argument("--width-scale", type=float, default=0.62,
                         help="عرض طرح نسبت به فاصله دو شانه (پیش‌فرض 0.62)")
    parser.add_argument("--height-scale", type=float, default=None,
                         help="نسبت ارتفاع به عرض طرح (پیش‌فرض: نسبت طبیعی خود PNG)")
    parser.add_argument("--vert-offset", type=float, default=0.35,
                         help="فاصله عمودی بالای طرح از خط شانه (پیش‌فرض 0.35)")
    parser.add_argument("--blend", type=float, default=0.35,
                         help="شدت افکت سایه/چروک پارچه روی طرح، 0 تا 1 (پیش‌فرض 0.35)")
    parser.add_argument("--smoothing", type=float, default=0.6,
                         help="میزان صاف‌سازی حرکت بین فریم‌ها، 0 تا 0.95 (پیش‌فرض 0.6)")
    parser.add_argument("--model", choices=["lite", "full"], default="lite",
                         help="نسخه مدل تشخیص بدن: lite (سریع‌تر) یا full (دقیق‌تر) - پیش‌فرض lite")

    args = parser.parse_args()

    process_video(
        video_path=args.video,
        design_path=args.design,
        output_path=args.output,
        width_scale=args.width_scale,
        height_scale=args.height_scale,
        vert_offset=args.vert_offset,
        blend_strength=args.blend,
        smoothing=args.smoothing,
        model_variant=args.model,
    )
    print(f"تمام شد. خروجی ذخیره شد در: {args.output}")


if __name__ == "__main__":
    main()
