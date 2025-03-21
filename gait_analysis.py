import time
import numpy as np

# 从config模块导入常量
from config import LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX, LEFT_KNEE_IDX, RIGHT_KNEE_IDX, LEFT_HIP_IDX, RIGHT_HIP_IDX, LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX, NOSE_IDX, STEP_WIDTH_RANGES, STEP_LENGTH_RANGES, STRIDE_LENGTH_RANGES, CADENCE_RANGES, STEP_LENGTH_SYMMETRY_RANGES, SUPPORT_TIME_DIFF_RANGES, SWING_TIME_DIFF_RANGES, PELVIC_ROTATION_RANGES, KNEE_FLEXION_RANGES, ANKLE_FLEXION_RANGES, WEIGHT_SHIFT_RANGES, get_severity_level

def calculate_angle(p1, p2, p3=None):
    """计算两点或三点之间的角度"""
    if p3 is None:
        # 计算与垂直线的角度
        return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    else:
        # 计算三点角度
        v1 = p1 - p2
        v2 = p3 - p2
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return angle

def calculate_distance(p1, p2):
    """计算两点之间的欧几里得距离"""
    return np.sqrt(np.sum((p2 - p1) ** 2))

def analyze_gait_metrics(keypoints, keypoint_scores, prev_frame_data=None):
    """分析步态指标"""
    # 关键点索引定义
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    results = {}
    keypoints = np.array(keypoints)

    # 计算步宽 (左右脚踝之间的距离)
    if keypoint_scores[LEFT_ANKLE] > 0.3 and keypoint_scores[RIGHT_ANKLE] > 0.3:
        step_width = calculate_distance(keypoints[LEFT_ANKLE], keypoints[RIGHT_ANKLE])
        results['step_width'] = step_width
        results['step_width_status'] = get_severity_level(step_width, STEP_WIDTH_RANGES)

    # 计算步长 (前后脚之间的距离)
    if prev_frame_data and all(v is not None for v in prev_frame_data.values()):
        left_step = abs(keypoints[LEFT_ANKLE][1] - prev_frame_data['left_ankle_y'])
        right_step = abs(keypoints[RIGHT_ANKLE][1] - prev_frame_data['right_ankle_y'])
        
        results['left_step_length'] = left_step
        results['right_step_length'] = right_step
        results['step_length_status'] = get_severity_level((left_step + right_step) / 2, STEP_LENGTH_RANGES)
        
        # 计算步长对称性
        step_diff = abs(left_step - right_step)
        results['step_symmetry'] = step_diff
        results['step_symmetry_status'] = get_severity_level(step_diff, STEP_LENGTH_SYMMETRY_RANGES)

    # 计算骨盆旋转角度
    if keypoint_scores[LEFT_HIP] > 0.3 and keypoint_scores[RIGHT_HIP] > 0.3:
        pelvic_angle = calculate_angle(keypoints[LEFT_HIP], keypoints[RIGHT_HIP])
        results['pelvic_rotation'] = pelvic_angle
        results['pelvic_rotation_status'] = get_severity_level(abs(pelvic_angle), PELVIC_ROTATION_RANGES)

    # 计算膝关节角度
    if all(keypoint_scores[[LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]] > 0.3):
        left_knee_angle = calculate_angle(keypoints[LEFT_HIP], keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE])
        results['left_knee_angle'] = left_knee_angle
        results['left_knee_status'] = get_severity_level(abs(left_knee_angle - 180), KNEE_FLEXION_RANGES)

    if all(keypoint_scores[[RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]] > 0.3):
        right_knee_angle = calculate_angle(keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE], keypoints[RIGHT_ANKLE])
        results['right_knee_angle'] = right_knee_angle
        results['right_knee_status'] = get_severity_level(abs(right_knee_angle - 180), KNEE_FLEXION_RANGES)

    # 计算踝关节角度
    if all(keypoint_scores[[LEFT_KNEE, LEFT_ANKLE]] > 0.3):
        left_ankle_angle = calculate_angle(keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE])
        results['left_ankle_angle'] = left_ankle_angle
        results['left_ankle_status'] = get_severity_level(abs(left_ankle_angle - 90), ANKLE_FLEXION_RANGES)

    if all(keypoint_scores[[RIGHT_KNEE, RIGHT_ANKLE]] > 0.3):
        right_ankle_angle = calculate_angle(keypoints[RIGHT_KNEE], keypoints[RIGHT_ANKLE])
        results['right_ankle_angle'] = right_ankle_angle
        results['right_ankle_status'] = get_severity_level(abs(right_ankle_angle - 90), ANKLE_FLEXION_RANGES)

    # 计算身体重心转移
    if all(keypoint_scores[[LEFT_HIP, RIGHT_HIP]] > 0.3):
        hip_center = (keypoints[LEFT_HIP] + keypoints[RIGHT_HIP]) / 2
        if keypoint_scores[LEFT_ANKLE] > 0.3 and keypoint_scores[RIGHT_ANKLE] > 0.3:
            ankle_center = (keypoints[LEFT_ANKLE] + keypoints[RIGHT_ANKLE]) / 2
            weight_shift = calculate_distance(hip_center, ankle_center)
            step_length = calculate_distance(keypoints[LEFT_ANKLE], keypoints[RIGHT_ANKLE])
            if step_length > 0:
                weight_shift_percent = (weight_shift / step_length) * 100
                results['weight_shift'] = weight_shift_percent
                results['weight_shift_status'] = get_severity_level(weight_shift_percent, WEIGHT_SHIFT_RANGES)

    return results

def calculate_gait_symmetry(left_time, right_time):
    """计算步态对称性百分比
    
    Args:
        left_time: 左腿时间
        right_time: 右腿时间
    
    Returns:
        float: 对称性百分比 (0-100)，50%表示完全对称
    """
    if left_time <= 0 or right_time <= 0:
        return 50  # 默认对称
    
    total = left_time + right_time
    left_percentage = (left_time / total) * 100
    
    # 50%表示完美对称
    # 返回一个0-100的值，其中50表示完全对称
    # 0表示完全右偏，100表示完全左偏
    return left_percentage

def generate_gait_summary(gait_metrics, gait_normal_ranges):
    """生成步态分析摘要
    
    Args:
        gait_metrics: 步态指标数据
        gait_normal_ranges: 正常范围参考值
    
    Returns:
        str: 步态分析摘要HTML
    """
    # 计算对称性
    symmetry = calculate_gait_symmetry(
        gait_metrics['左腿抬起时间'], 
        gait_metrics['右腿抬起时间']
    )
    
    # 确定对称性的描述
    symmetry_percentage = abs(symmetry - 50) * 2  # 0-100%，0%是完全对称
    if symmetry_percentage < 10:
        symmetry_text = "步态对称性良好"
        symmetry_advice = "继续保持当前步态节奏"
    elif symmetry_percentage < 20:
        symmetry_text = "步态轻微不对称"
        symmetry_advice = "建议注意平衡左右腿的运动"
    else:
        symmetry_text = "步态明显不对称"
        left_or_right = "左" if symmetry > 50 else "右"
        symmetry_advice = f"存在{left_or_right}侧偏好，建议调整步态平衡"
    
    # 生成对称性指示器的位置 (0-100%)
    marker_position = symmetry
    
    # 检查步时是否在正常范围内
    step_time = gait_metrics['步时']
    min_step, max_step = gait_normal_ranges['步时']
    
    if step_time < min_step:
        pace_text = "步频偏快"
        pace_advice = "可以适当放慢步频，增加稳定性"
    elif step_time > max_step:
        pace_text = "步频偏慢"
        pace_advice = "可以适当增加步频，提高活动效率"
    else:
        pace_text = "步频正常"
        pace_advice = "当前步频处于健康范围"
    
    # 生成总结HTML
    summary_html = f"""
    <div class="gait-summary">
        <strong>步态分析总结：</strong><br>
        1. {symmetry_text}：{symmetry_advice}<br>
        <div class="symmetry-label">
            <span>右腿偏好</span>
            <span>平衡</span>
            <span>左腿偏好</span>
        </div>
        <div class="symmetry-indicator">
            <div class="symmetry-marker" style="left: {marker_position}%;"></div>
        </div>
        <br>
        2. {pace_text}：{pace_advice}<br>
        步时: {round(step_time, 2)}秒 (正常范围: {min_step}-{max_step}秒)
    </div>
    """
    
    return summary_html

def display_gait_metric(title, value, unit="秒"):
    """显示单个步态指标的卡片
    
    Args:
        title: 指标标题
        value: 指标值
        unit: 单位，默认为"秒"
        
    Returns:
        str: HTML格式的指标卡片
    """
    value_rounded = round(value, 2) if value is not None else 0
    return f"""<div class="gait-metric-card">
    <div class="gait-metric-title">{title}</div>
    <div class="gait-metric-value">{value_rounded} {unit}</div>
</div>"""

def init_gait_history():
    """初始化步态历史数据
    
    Returns:
        dict: 包含空步态历史数据的字典
    """
    return {
        '左腿抬起时间': [],
        '右腿抬起时间': [],
        '双支撑时间': [],
        '步时': [],
        '摆动时间': [],
        '支撑时间': []
    }

def update_gait_history(gait_history, gait_metrics, max_history_points=100):
    """更新步态历史数据
    
    Args:
        gait_history: 现有的步态历史数据
        gait_metrics: 当前帧的步态指标
        max_history_points: 最大历史数据点数量，默认为100
        
    Returns:
        dict: 更新后的步态历史数据
    """
    for metric, value in gait_metrics.items():
        gait_history[metric].append(value)
        # 限制历史数据量
        if len(gait_history[metric]) > max_history_points:
            gait_history[metric] = gait_history[metric][-max_history_points:]
    
    return gait_history 