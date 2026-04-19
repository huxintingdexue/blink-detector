import gc
from dataclasses import dataclass
from io import BytesIO

import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image


LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
DEFAULT_EAR_THRESHOLD = 0.2
RESIZE_MAX_SIDE = 1600
MIN_EYE_SPAN_PX = 4.0


@dataclass
class FaceSummary:
    status: str
    blink: bool = False


@st.cache_resource
def load_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=100,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def resize_image(image: Image.Image, max_side: int = RESIZE_MAX_SIDE) -> Image.Image:
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_side:
        return image.copy()

    scale = max_side / float(longest_side)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def normalized_to_pixel(landmark, width: int, height: int):
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def eye_points_valid(indices, landmarks, width: int, height: int):
    points = []
    for index in indices:
        if index >= len(landmarks):
            return False, []

        point = landmarks[index]
        if point.x is None or point.y is None:
            return False, []

        pixel_point = normalized_to_pixel(point, width, height)
        # Allow a small margin because normalized landmarks can slightly exceed bounds.
        if (
            pixel_point[0] < -width * 0.05
            or pixel_point[0] > width * 1.05
            or pixel_point[1] < -height * 0.05
            or pixel_point[1] > height * 1.05
        ):
            return False, []
        points.append(pixel_point)

    return True, points


def calc_ear(points) -> float:
    p1, p2, p3, p4, p5, p6 = points
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal <= 0:
        return 0.0

    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    return float(vertical / (2.0 * horizontal))


def classify_face(face_landmarks, width: int, height: int, ear_threshold: float) -> FaceSummary:
    landmarks = face_landmarks.landmark
    left_ok, left_points = eye_points_valid(LEFT_EYE_INDICES, landmarks, width, height)
    right_ok, right_points = eye_points_valid(RIGHT_EYE_INDICES, landmarks, width, height)

    if not left_ok and not right_ok:
        return FaceSummary(status="invalid")

    if not left_ok or not right_ok:
        return FaceSummary(status="uncertain")

    left_span = np.linalg.norm(left_points[0] - left_points[3])
    right_span = np.linalg.norm(right_points[0] - right_points[3])
    if left_span < MIN_EYE_SPAN_PX or right_span < MIN_EYE_SPAN_PX:
        return FaceSummary(status="uncertain")

    left_ear = calc_ear(left_points)
    right_ear = calc_ear(right_points)
    blink = left_ear < ear_threshold or right_ear < ear_threshold
    return FaceSummary(status="valid", blink=blink)


def make_thumbnail(image: Image.Image, size=(180, 180)) -> Image.Image:
    thumbnail = image.copy()
    thumbnail.thumbnail(size)
    return thumbnail


def process_uploaded_file(uploaded_file, face_mesh, ear_threshold: float):
    raw_bytes = uploaded_file.getvalue()
    with Image.open(BytesIO(raw_bytes)) as source_image:
        resized_image = resize_image(source_image)
        rgb_array = pil_to_rgb_array(resized_image)

    results = face_mesh.process(rgb_array)
    face_landmarks_list = results.multi_face_landmarks or []
    height, width = rgb_array.shape[:2]

    valid_faces = 0
    uncertain_faces = 0
    invalid_faces = 0
    blink_count = 0

    for face_landmarks in face_landmarks_list:
        summary = classify_face(face_landmarks, width, height, ear_threshold)
        if summary.status == "valid":
            valid_faces += 1
            if summary.blink:
                blink_count += 1
        elif summary.status == "uncertain":
            uncertain_faces += 1
        else:
            invalid_faces += 1

    thumbnail = make_thumbnail(resized_image)

    result = {
        "file_name": uploaded_file.name,
        "file_bytes": raw_bytes,
        "mime_type": uploaded_file.type or "application/octet-stream",
        "blink_count": blink_count,
        "valid_faces": valid_faces,
        "uncertain_faces": uncertain_faces,
        "invalid_faces": invalid_faces,
        "total_detected_faces": len(face_landmarks_list),
        "thumbnail": thumbnail,
    }

    del rgb_array
    del results
    gc.collect()
    return result


def get_upload_signature(files, ear_threshold: float) -> tuple:
    return tuple((file.name, file.size) for file in files) + (round(ear_threshold, 4),)


def render_recommendation(best_result):
    st.subheader("推荐结果")
    st.success(
        f"✅ 推荐选择：{best_result['file_name']}  |  闭眼人数 {best_result['blink_count']}"
    )
    cols = st.columns([1.4, 1])
    with cols[0]:
        st.image(best_result["file_bytes"], use_container_width=True)
    with cols[1]:
        st.metric("闭眼人数", best_result["blink_count"])
        st.write(f"有效人脸数：{best_result['valid_faces']}")
        st.write(f"不确定人脸数：{best_result['uncertain_faces']}")
        st.write(f"无效人脸数：{best_result['invalid_faces']}")
        st.download_button(
            "下载推荐照片（原图）",
            data=best_result["file_bytes"],
            file_name=best_result["file_name"],
            mime=best_result["mime_type"],
            use_container_width=True,
            key=f"download-best-{best_result['file_name']}",
            type="primary",
        )


def render_result_list(results):
    st.subheader("结果列表")
    header = st.columns([1.2, 2.2, 1, 1, 1, 1, 1.2])
    header[0].markdown("**缩略图**")
    header[1].markdown("**文件名**")
    header[2].markdown("**闭眼人数**")
    header[3].markdown("**有效**")
    header[4].markdown("**不确定**")
    header[5].markdown("**无效**")
    header[6].markdown("**导出**")

    for item in results:
        cols = st.columns([1.2, 2.2, 1, 1, 1, 1, 1.2])
        cols[0].image(item["thumbnail"], use_container_width=True)
        cols[1].write(item["file_name"])
        cols[2].write(item["blink_count"])
        cols[3].write(item["valid_faces"])
        cols[4].write(item["uncertain_faces"])
        cols[5].write(item["invalid_faces"])
        cols[6].download_button(
            "下载原图",
            data=item["file_bytes"],
            file_name=item["file_name"],
            mime=item["mime_type"],
            key=f"download-{item['file_name']}",
            use_container_width=True,
        )


def ensure_state():
    st.session_state.setdefault("last_signature", None)
    st.session_state.setdefault("results", [])


def process_if_needed(uploaded_files, face_mesh, ear_threshold: float):
    signature = get_upload_signature(uploaded_files, ear_threshold)
    if st.session_state["last_signature"] == signature and st.session_state["results"]:
        return st.session_state["results"]

    progress_bar = st.progress(0, text="正在分析照片...")
    status = st.empty()
    processed_results = []
    total_files = len(uploaded_files)

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        status.write(f"处理中：{uploaded_file.name} ({index}/{total_files})")
        processed_results.append(process_uploaded_file(uploaded_file, face_mesh, ear_threshold))
        progress_bar.progress(index / total_files, text=f"已完成 {index}/{total_files}")

    processed_results.sort(
        key=lambda item: (
            item["blink_count"],
            -item["valid_faces"],
            item["uncertain_faces"],
            item["file_name"].lower(),
        )
    )

    st.session_state["last_signature"] = signature
    st.session_state["results"] = processed_results
    status.empty()
    progress_bar.empty()
    return processed_results


def main():
    st.set_page_config(page_title="照片闭眼检测工具", layout="wide")
    ensure_state()

    st.title("照片闭眼检测工具")
    st.caption("上传同一场景的多张照片，自动统计每张照片中的闭眼人数并推荐最佳结果。")

    with st.sidebar:
        st.header("参数")
        ear_threshold = st.slider(
            "EAR 闭眼阈值",
            min_value=0.10,
            max_value=0.35,
            value=DEFAULT_EAR_THRESHOLD,
            step=0.01,
            help="默认使用 0.20。EAR 越小，越可能是闭眼。",
        )

    uploaded_files = st.file_uploader(
        "上传照片（支持 JPG / PNG，可一次选择多张）",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.session_state["last_signature"] = None
        st.session_state["results"] = []
        st.info("请先上传一组同一场景的照片。")
        return

    face_mesh = load_face_mesh()
    results = process_if_needed(uploaded_files, face_mesh, ear_threshold)

    if not results:
        st.warning("未检测到可用结果，请更换图片后重试。")
        return

    render_recommendation(results[0])
    render_result_list(results)


if __name__ == "__main__":
    main()
