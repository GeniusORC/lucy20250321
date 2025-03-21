import streamlit as st
from config import NORMAL_RANGES

def check_value_in_range(value, range_str):
    """检查值是否在正常范围内
    
    Args:
        value: 需要检查的值
        range_str: 正常范围字符串，格式如"0°～5°", ">65°", "<39°"等
    
    Returns:
        bool: 值是否在正常范围内
    """
    if value is None:
        return False
    
    if '～' in range_str:
        parts = range_str.replace('°', '').split('～')
        min_val = float(parts[0].replace('%', ''))
        max_val = float(parts[1].replace('%', ''))
        return min_val <= value <= max_val
    elif '<' in range_str:
        max_val = float(range_str.replace('<', '').replace('°', '').replace('%', ''))
        return value < max_val
    elif '>' in range_str:
        min_val = float(range_str.replace('>', '').replace('°', '').replace('%', ''))
        return value > min_val
    return False

def display_metric(title, value, normal_range, key):
    """显示单个指标的卡片
    
    Args:
        title: 指标标题
        value: 指标值
        normal_range: 正常范围字符串
        key: 用于唯一标识该指标的键
    """
    if value is not None:
        unit = '%' if '肥胖度' in title else '°'
        is_normal = check_value_in_range(value, normal_range)
        status_class = "metric-normal" if is_normal else "metric-warning"
        badge_class = "normal-badge" if is_normal else "warning-badge"
        status_text = "正常" if is_normal else "异常"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 class="metric-title">{title}</h4>
            <p class="metric-value">
                <span class="value-badge {badge_class}">{value}{unit}</span>
                <span class="{status_class}">({status_text})</span> | 范围: {normal_range}
            </p>
        </div>
        """, unsafe_allow_html=True)

def generate_posture_summary(posture_results):
    """生成姿态分析摘要
    
    Args:
        posture_results: 姿态分析结果字典
    
    Returns:
        str: 姿态分析摘要HTML
        list: 异常指标列表
    """
    # 计算异常指标数量
    abnormal_metrics = []
    for title, value in posture_results.items():
        if value is not None and not check_value_in_range(value, NORMAL_RANGES[title]):
            abnormal_metrics.append(title)
    
    # 显示总结
    total_metrics = sum(1 for v in posture_results.values() if v is not None)
    
    if total_metrics > 0:
        abnormal_count = len(abnormal_metrics)
        normal_count = total_metrics - abnormal_count
        
        if abnormal_count > 0:
            advice = "建议关注以下异常指标并进行相应的调整和训练。"
            abnormal_text = "、".join(abnormal_metrics[:3])
            if len(abnormal_metrics) > 3:
                abnormal_text += f"等{len(abnormal_metrics)}项"
        else:
            advice = "您的体态状况良好，请继续保持。"
            abnormal_text = ""
        
        summary = f"""<div class="metrics-summary">
            检测到{total_metrics}项指标，其中{normal_count}项正常，{abnormal_count}项异常。{advice}
            {f'<br><span class="metric-warning">异常项: {abnormal_text}</span>' if abnormal_count > 0 else ''}
        </div>"""
        
        return summary, abnormal_metrics
    
    return "<div class='metrics-summary'>未检测到有效姿态指标</div>", []

def display_posture_metrics(posture_results):
    """显示按组分类的姿态指标
    
    Args:
        posture_results: 姿态分析结果字典
    """
    # 按分类组织指标
    metrics_grouped = {
        "头部": ["头前倾角", "头侧倾角", "头旋转角"],
        "上半身": ["肩倾斜角", "圆肩角", "背部角"],
        "中部": ["腹部肥胖度", "腰曲度", "骨盆前倾角", "侧中位度"],
        "下肢": ["腿型-左腿", "腿型-右腿", "左膝评估角", "右膝评估角", "身体倾斜度", "足八角"]
    }
    
    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
    st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
    
    # 显示按组分类的指标
    for group, metrics in metrics_grouped.items():
        metrics_in_group = [m for m in metrics if m in posture_results and posture_results[m] is not None]
        if metrics_in_group:
            st.markdown(f'<div class="metrics-group-title">{group}</div>', unsafe_allow_html=True)
            for metric in metrics_in_group:
                display_metric(metric, posture_results[metric], NORMAL_RANGES[metric], metric)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) 