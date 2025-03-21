import cv2
import numpy as np
import time
import mmcv
from mmpose.registry import VISUALIZERS
from mmpose.utils import adapt_mmdet_pipeline
import streamlit as st

# 初始化检测器
def init_detector(config, checkpoint, device):
    """初始化物体检测器"""
    from mmdet.apis import init_detector as init_det
    detector = init_det(config, checkpoint, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    return detector

# 初始化姿态估计器
def init_pose_estimator(config, checkpoint, device, cfg_options=None):
    """初始化姿态估计器"""
    from mmpose.apis import init_model
    pose_estimator = init_model(
        config, 
        checkpoint, 
        device=device, 
        cfg_options=cfg_options, 
        revision='v1.0.0',  # 使用v1.0.0版本的API
        strict=False  # 允许部分权重不匹配
    )
    return pose_estimator

# 处理单个图像，返回姿态预测结果
def process_one_image(args, img, detector, pose_estimator, visualizer, delay=0.001):
    """处理单个图像并返回姿态预测结果"""
    from webcam_rtmw_demo import analyze_body_posture, filter_keypoints

    # 复制原始图像用于可视化
    img_vis = img.copy()
    
    # 确保visualizer设置了图像
    if visualizer is not None:
        visualizer.set_image(img_vis)
    
    # 检测目标对象
    from mmdet.apis import inference_detector
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    
    # 过滤检测结果
    bboxes = pred_instance.bboxes
    scores = pred_instance.scores
    labels = pred_instance.labels
    
    # 获取物体检测结果
    if len(bboxes) > 0:
        # 选择检测分数最高的人物
        person_indices = np.where(labels == args.det_cat_id)[0]
        if len(person_indices) > 0:
            best_idx = person_indices[scores[person_indices].argmax()]
            bbox = bboxes[best_idx]
            score = scores[best_idx]
            
            # 分数低于阈值，跳过
            if score < args.bbox_thr:
                return None
            
            # 进行姿态估计
            from mmpose.apis import inference_topdown
            pose_results = inference_topdown(pose_estimator, img, [{"bbox": bbox}])
            
            # 过滤关键点
            pose_results[0] = filter_keypoints(pose_results[0], args)
            
            # 准备可视化数据
            data_samples = pose_results[0].get('pred_instances', None)
            
            # 分析体态数据
            keypoints = data_samples.keypoints[0]
            keypoint_scores = data_samples.keypoint_scores[0]
            
            # 分析和添加自定义关键点
            custom_keypoints = None
            if args.draw_iliac_midpoint or args.draw_neck_midpoint:
                custom_keypoints = analyze_body_posture(keypoints, keypoint_scores, return_keypoints=True)
                pose_results[0]['custom_keypoints'] = custom_keypoints
            
            # 可视化结果
            if args.show_posture_analysis and visualizer is not None:
                # 绘制边界框
                if args.draw_bbox:
                    visualizer.draw_bboxes(np.array([bbox]), edge_colors='blue', alpha=0.5)
                
                # 绘制关键点和骨架
                if args.draw_keypoints:
                    visualizer.draw_instance_keypoints(data_samples)
                
                # 绘制自定义关键点
                if (hasattr(pose_results[0], 'custom_keypoints') and 
                    custom_keypoints is not None and 
                    len(custom_keypoints) > 0):
                    # 获取自定义关键点
                    image = visualizer.get_image()
                    
                    # 绘制髂骨中点 (ILIAC_MIDPOINT_IDX = 0)
                    if args.draw_iliac_midpoint and 0 < len(custom_keypoints):
                        pt = custom_keypoints[0]
                        x, y = int(pt[0]), int(pt[1])
                        cv2.circle(image, (x, y), args.custom_keypoint_radius, 
                                  (0, 255, 0), -1)  # 绿色填充圆
                        
                        # 连接到左右髋关键点
                        if keypoint_scores[11] > 0.5 and keypoint_scores[12] > 0.5:  # 左右髋
                            left_hip = keypoints[11]
                            right_hip = keypoints[12]
                            left_x, left_y = int(left_hip[0]), int(left_hip[1])
                            right_x, right_y = int(right_hip[0]), int(right_hip[1])
                            cv2.line(image, (x, y), (left_x, left_y), (0, 255, 0), 
                                    args.custom_keypoint_thickness)
                            cv2.line(image, (x, y), (right_x, right_y), (0, 255, 0), 
                                    args.custom_keypoint_thickness)
                    
                    # 绘制颈椎中点 (NECK_MIDPOINT_IDX = 1)
                    if args.draw_neck_midpoint and 1 < len(custom_keypoints):
                        pt = custom_keypoints[1]
                        x, y = int(pt[0]), int(pt[1])
                        cv2.circle(image, (x, y), args.custom_keypoint_radius, 
                                  (0, 0, 255), -1)  # 红色填充圆
                        
                        # 连接到鼻子关键点
                        if keypoint_scores[0] > 0.5:  # 鼻子
                            nose = keypoints[0]
                            nose_x, nose_y = int(nose[0]), int(nose[1])
                            cv2.line(image, (x, y), (nose_x, nose_y), (0, 0, 255), 
                                    args.custom_keypoint_thickness)
                    
                    # 更新可视化器的图像
                    visualizer.set_image(image)
            
            # 添加延迟以减缓处理速度
            if delay > 0:
                time.sleep(delay)
            
            return pose_results[0]
    
    return None

# 缓存加载模型函数
@st.cache_resource
def load_models(_args):
    """加载模型并返回检测器、姿态估计器和可视化器"""
    # 初始化检测器
    detector = init_detector(
        _args.det_config, _args.det_checkpoint, device=_args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    
    # 初始化姿态估计器
    pose_estimator = init_pose_estimator(
        _args.pose_config,
        _args.pose_checkpoint,
        device=_args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=_args.draw_heatmap))))
    
    # 配置可视化设置
    pose_estimator.cfg.visualizer.radius = _args.radius
    pose_estimator.cfg.visualizer.alpha = _args.alpha
    pose_estimator.cfg.visualizer.line_width = _args.thickness
    
    # 创建可视化器
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=_args.skeleton_style)
    
    return detector, pose_estimator, visualizer 