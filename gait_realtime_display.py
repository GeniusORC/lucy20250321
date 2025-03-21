import cv2
import numpy as np
import os
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.structures import merge_data_samples
from gait_analysis import analyze_gait_metrics
from config import (
    STEP_WIDTH_RANGES, STEP_LENGTH_RANGES, STRIDE_LENGTH_RANGES,
    CADENCE_RANGES, STEP_LENGTH_SYMMETRY_RANGES, SUPPORT_TIME_DIFF_RANGES,
    SWING_TIME_DIFF_RANGES, PELVIC_ROTATION_RANGES, KNEE_FLEXION_RANGES,
    ANKLE_FLEXION_RANGES, WEIGHT_SHIFT_RANGES, get_severity_level
)
from PIL import Image, ImageDraw, ImageFont

# 定义骨架连接
SKELETON_CONNECTIONS = [
    # 躯干
    (5, 6),   # 左肩 -> 右肩
    (5, 11),  # 左肩 -> 左髋
    (6, 12),  # 右肩 -> 右髋
    (11, 12), # 左髋 -> 右髋
    
    # 左臂
    (5, 7),   # 左肩 -> 左肘
    (7, 9),   # 左肘 -> 左腕
    
    # 右臂
    (6, 8),   # 右肩 -> 右肘
    (8, 10),  # 右肘 -> 右腕
    
    # 左腿
    (11, 13), # 左髋 -> 左膝
    (13, 15), # 左膝 -> 左踝
    
    # 右腿
    (12, 14), # 右髋 -> 右膝
    (14, 16), # 右膝 -> 右踝
    
    # 头部连接
    (0, 5),   # 鼻子 -> 左肩
    (0, 6),   # 鼻子 -> 右肩
]

def is_in_range(value, range_tuple):
    """检查值是否在正常范围内"""
    if value is None:
        return False
    return range_tuple[0] <= value <= range_tuple[1]

def format_result(name, value, normal_range):
    """格式化结果输出"""
    if value is None:
        return None
    status = "正常" if is_in_range(value, normal_range) else "异常"
    return f"{name:<12} {value:>6.1f}° {normal_range[0]}°～{normal_range[1]}° {status}"

def draw_skeleton(frame, keypoints, keypoint_scores, threshold=0.3):
    """绘制骨架连接"""
    # 关键点显示参数
    point_color = (0, 255, 0)  # 绿色
    point_size = 6  # 关键点大小
    point_thickness = -1  # 实心圆
    
    # 连接线显示参数
    line_color = (0, 255, 255)  # 黄色
    line_thickness = 2

    # 首先绘制所有连接线
    for connection in SKELETON_CONNECTIONS:
        start_idx, end_idx = connection
        if (keypoint_scores[start_idx] > threshold and 
            keypoint_scores[end_idx] > threshold):
            start_point = tuple(map(int, keypoints[start_idx]))
            end_point = tuple(map(int, keypoints[end_idx]))
            cv2.line(frame, start_point, end_point, line_color, line_thickness)

    # 然后绘制关键点（这样关键点会显示在连接线上面）
    for i, (x, y) in enumerate(keypoints):
        if keypoint_scores[i] > threshold:
            point = (int(x), int(y))
            cv2.circle(frame, point, point_size, point_color, point_thickness)

def put_chinese_text(img, text, position, text_color, font_size=20):
    """在图片上添加中文文本"""
    # 创建一个支持中文的图片
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    # 加载中文字体
    try:
        # 尝试多个可能的字体路径
        font_paths = [
            "simhei.ttf",
            "NotoSansCJK-Regular.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simkai.ttf"
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        if font is None:
            raise Exception("未找到可用的中文字体")
    except Exception as e:
        print(f"警告：{str(e)}，将使用默认字体")
        font = ImageFont.load_default()
    
    # 绘制文本
    draw.text(position, text, font=font, fill=text_color)
    
    # 转换回OpenCV格式
    return np.array(img_pil)

def calculate_score(gait_results):
    """计算步态评分"""
    scores = {}
    total_score = 0
    max_score = 68  # 总分68分

    # 步宽评分（10分）
    if 'step_width' in gait_results:
        if gait_results['step_width_status'] == "正常":
            scores['step_width'] = 10
        elif gait_results['step_width_status'] == "轻度异常":
            scores['step_width'] = 7
        elif gait_results['step_width_status'] == "中度异常":
            scores['step_width'] = 4
        else:
            scores['step_width'] = 0
        total_score += scores['step_width']

    # 步长评分（5分）
    if 'left_step_length' in gait_results and 'right_step_length' in gait_results:
        if gait_results['step_length_status'] == "正常":
            scores['step_length'] = 5
        elif gait_results['step_length_status'] == "轻度异常":
            scores['step_length'] = 3
        elif gait_results['step_length_status'] == "中度异常":
            scores['step_length'] = 1
        else:
            scores['step_length'] = 0
        total_score += scores['step_length']

    # 步长对称性评分（5分）
    if 'step_symmetry' in gait_results:
        if gait_results['step_symmetry_status'] == "正常":
            scores['step_symmetry'] = 5
        elif gait_results['step_symmetry_status'] == "轻度异常":
            scores['step_symmetry'] = 3
        elif gait_results['step_symmetry_status'] == "中度异常":
            scores['step_symmetry'] = 1
        else:
            scores['step_symmetry'] = 0
        total_score += scores['step_symmetry']

    # 骨盆旋转评分（5分）
    if 'pelvic_rotation' in gait_results:
        if gait_results['pelvic_rotation_status'] == "正常":
            scores['pelvic_rotation'] = 5
        elif gait_results['pelvic_rotation_status'] == "轻度异常":
            scores['pelvic_rotation'] = 3
        elif gait_results['pelvic_rotation_status'] == "中度异常":
            scores['pelvic_rotation'] = 1
        else:
            scores['pelvic_rotation'] = 0
        total_score += scores['pelvic_rotation']

    # 膝关节角度评分（3分）
    if 'left_knee_angle' in gait_results and 'right_knee_angle' in gait_results:
        knee_score = 0
        if gait_results['left_knee_status'] == "正常":
            knee_score += 1.5
        elif gait_results['left_knee_status'] == "轻度异常":
            knee_score += 1
        elif gait_results['left_knee_status'] == "中度异常":
            knee_score += 0.5
        
        if gait_results['right_knee_status'] == "正常":
            knee_score += 1.5
        elif gait_results['right_knee_status'] == "轻度异常":
            knee_score += 1
        elif gait_results['right_knee_status'] == "中度异常":
            knee_score += 0.5
        
        scores['knee_angle'] = knee_score
        total_score += knee_score

    # 踝关节角度评分（1分）
    if 'left_ankle_angle' in gait_results and 'right_ankle_angle' in gait_results:
        ankle_score = 0
        if gait_results['left_ankle_status'] == "正常":
            ankle_score += 0.5
        elif gait_results['left_ankle_status'] == "轻度异常":
            ankle_score += 0.3
        elif gait_results['left_ankle_status'] == "中度异常":
            ankle_score += 0.1
        
        if gait_results['right_ankle_status'] == "正常":
            ankle_score += 0.5
        elif gait_results['right_ankle_status'] == "轻度异常":
            ankle_score += 0.3
        elif gait_results['right_ankle_status'] == "中度异常":
            ankle_score += 0.1
        
        scores['ankle_angle'] = ankle_score
        total_score += ankle_score

    # 重心转移评分（10分）
    if 'weight_shift' in gait_results:
        if gait_results['weight_shift_status'] == "正常":
            scores['weight_shift'] = 10
        elif gait_results['weight_shift_status'] == "轻度异常":
            scores['weight_shift'] = 7
        elif gait_results['weight_shift_status'] == "中度异常":
            scores['weight_shift'] = 4
        else:
            scores['weight_shift'] = 0
        total_score += scores['weight_shift']

    return scores, total_score, max_score

def draw_analysis_results(frame, gait_results):
    """在图像上绘制分析结果"""
    # 获取图像尺寸
    height, width = frame.shape[:2]
    
    # 设置文本参数
    font_size = int(height * 0.025)  # 稍微减小字体大小以适应更多内容
    text_color = (255, 255, 255)  # 白色
    padding = int(width * 0.02)  # 根据图像宽度调整内边距
    line_height = int(height * 0.035)  # 稍微减小行高以适应更多内容

    # 创建半透明黑色背景
    overlay = frame.copy()
    panel_width = int(width * 0.3)  # 增加面板宽度以适应评分
    cv2.rectangle(overlay, (0, 0), (panel_width, height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # 计算评分
    scores, total_score, max_score = calculate_score(gait_results)

    # 准备显示的文本
    texts = []
    if 'step_width' in gait_results:
        status_color = (0, 255, 0) if gait_results['step_width_status'] == "正常" else (0, 0, 255)
        score = scores.get('step_width', 0)
        texts.append((f"步宽: {gait_results['step_width']:.1f}cm", gait_results['step_width_status'], status_color, f"{score}/10分"))
    
    if 'left_step_length' in gait_results and 'right_step_length' in gait_results:
        avg_step_length = (gait_results['left_step_length'] + gait_results['right_step_length']) / 2
        status = gait_results.get('step_length_status', '未知')
        status_color = (0, 255, 0) if status == "正常" else (0, 0, 255)
        score = scores.get('step_length', 0)
        texts.append((f"步长: {avg_step_length:.1f}cm", status, status_color, f"{score}/5分"))
    
    if 'step_symmetry' in gait_results:
        status_color = (0, 255, 0) if gait_results['step_symmetry_status'] == "正常" else (0, 0, 255)
        score = scores.get('step_symmetry', 0)
        texts.append((f"步长对称性: {gait_results['step_symmetry']:.1f}cm", gait_results['step_symmetry_status'], status_color, f"{score}/5分"))
    
    if 'pelvic_rotation' in gait_results:
        status_color = (0, 255, 0) if gait_results['pelvic_rotation_status'] == "正常" else (0, 0, 255)
        score = scores.get('pelvic_rotation', 0)
        texts.append((f"骨盆旋转: {gait_results['pelvic_rotation']:.1f}°", gait_results['pelvic_rotation_status'], status_color, f"{score}/5分"))
    
    if 'left_knee_angle' in gait_results:
        status_color = (0, 255, 0) if gait_results['left_knee_status'] == "正常" else (0, 0, 255)
        texts.append((f"左膝角度: {gait_results['left_knee_angle']:.1f}°", gait_results['left_knee_status'], status_color, ""))
    
    if 'right_knee_angle' in gait_results:
        status_color = (0, 255, 0) if gait_results['right_knee_status'] == "正常" else (0, 0, 255)
        score = scores.get('knee_angle', 0)
        texts.append((f"右膝角度: {gait_results['right_knee_angle']:.1f}°", gait_results['right_knee_status'], status_color, f"{score}/3分"))
    
    if 'left_ankle_angle' in gait_results:
        status_color = (0, 255, 0) if gait_results['left_ankle_status'] == "正常" else (0, 0, 255)
        texts.append((f"左踝角度: {gait_results['left_ankle_angle']:.1f}°", gait_results['left_ankle_status'], status_color, ""))
    
    if 'right_ankle_angle' in gait_results:
        status_color = (0, 255, 0) if gait_results['right_ankle_status'] == "正常" else (0, 0, 255)
        score = scores.get('ankle_angle', 0)
        texts.append((f"右踝角度: {gait_results['right_ankle_angle']:.1f}°", gait_results['right_ankle_status'], status_color, f"{score}/1分"))
    
    if 'weight_shift' in gait_results:
        status_color = (0, 255, 0) if gait_results['weight_shift_status'] == "正常" else (0, 0, 255)
        score = scores.get('weight_shift', 0)
        texts.append((f"重心转移: {gait_results['weight_shift']:.1f}%", gait_results['weight_shift_status'], status_color, f"{score}/10分"))

    # 绘制标题
    title = "步态分析结果"
    frame = put_chinese_text(frame, title, (padding, padding), text_color, font_size + 4)

    # 绘制结果
    y = padding + font_size + line_height
    for text, status, status_color, score in texts:
        # 绘制指标值
        frame = put_chinese_text(frame, text, (padding, y), text_color, font_size)
        
        # 绘制状态
        status_x = int(panel_width * 0.5)  # 调整状态文本的起始x坐标
        frame = put_chinese_text(frame, status, (status_x, y), status_color, font_size)
        
        # 绘制分数
        if score:
            score_x = int(panel_width * 0.75)  # 分数显示位置
            frame = put_chinese_text(frame, score, (score_x, y), text_color, font_size)
        
        y += line_height

    # 绘制总分
    total_score_text = f"总分: {total_score:.1f}/{max_score}分"
    frame = put_chinese_text(frame, total_score_text, (padding, y + line_height), text_color, font_size + 2)

    return frame

def main():
    # 初始化检测器
    det_config = 'projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py'
    det_checkpoint = 'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
    detector = init_detector(det_config, det_checkpoint)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # 初始化姿态估计器
    pose_config = 'configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py'
    pose_checkpoint = 'rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth'
    pose_estimator = init_pose_estimator(pose_config, pose_checkpoint)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    print("按 'q' 键退出程序")

    # 创建窗口并设置为全屏
    cv2.namedWindow('步态分析', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('步态分析', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_frame_data = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 调整图像大小以适应全屏
        screen_height = cv2.getWindowImageRect('步态分析')[3]
        screen_width = cv2.getWindowImageRect('步态分析')[2]
        frame = cv2.resize(frame, (screen_width, screen_height))

        # 检测人体
        det_result = inference_detector(detector, frame)
        
        # 确保检测到了人体
        if len(det_result.pred_instances.bboxes) == 0:
            # 即使没有检测到人体，也显示分析结果面板
            frame = draw_analysis_results(frame, {})
            cv2.imshow('步态分析', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 获取人体框并转换为CPU张量
        bboxes = det_result.pred_instances.bboxes.cpu()
        
        # 进行姿态估计
        pose_results = inference_topdown(pose_estimator, frame, bboxes)
        
        # 确保有姿态估计结果
        if not pose_results:
            # 即使没有姿态估计结果，也显示分析结果面板
            frame = draw_analysis_results(frame, {})
            cv2.imshow('步态分析', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            
        # 获取关键点和分数
        keypoints = pose_results[0].pred_instances.keypoints
        keypoint_scores = pose_results[0].pred_instances.keypoint_scores
        
        # 分析步态指标
        gait_results = analyze_gait_metrics(keypoints[0], keypoint_scores[0], prev_frame_data)
        
        # 更新前一帧数据
        prev_frame_data = {
            'left_knee_y': float(keypoints[0][13][1]) if keypoint_scores[0][13] > 0.3 else None,
            'right_knee_y': float(keypoints[0][14][1]) if keypoint_scores[0][14] > 0.3 else None,
            'left_ankle_y': float(keypoints[0][15][1]) if keypoint_scores[0][15] > 0.3 else None,
            'right_ankle_y': float(keypoints[0][16][1]) if keypoint_scores[0][16] > 0.3 else None
        }

        # 在图像上绘制骨架
        draw_skeleton(frame, keypoints[0], keypoint_scores[0])

        # 在图像上绘制分析结果
        frame = draw_analysis_results(frame, gait_results)

        # 显示图像
        cv2.imshow('步态分析', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 