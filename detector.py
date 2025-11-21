import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from tqdm import tqdm
import itertools
import math
from collections import deque

# === CONFIGURATION ===
DEVICE = torch.device('cpu')  # Только CPU для Docker
NUM_KEYPOINTS = 20
SEQ_LENGTH = 120

ALL_DIST_PAIRS = list(itertools.combinations(range(NUM_KEYPOINTS), 2))
ALL_ANGLE_TRIPLES = list(itertools.combinations(range(NUM_KEYPOINTS), 3))

# Загрузка моделей (относительные пути для Docker)
pose_model = YOLO("models/dog_pose_model_yolo8_14.pt")
dog_detect_model = YOLO("models/dog_detect_model_yolo8_450ep.pt")


class LSTMPoseClassifier(nn.Module):
    def __init__(self, input_size, lstm_hidden=256, num_lstm_layers=3, fc_layers=[512, 256]):
        super().__init__()
        # Замена GRU на LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.3 if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        # Механизм внимания (работает с выходами LSTM: [batch, seq, hidden*2])
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )
        # Глубокая полносвязная часть
        fc_modules = []
        in_features = lstm_hidden * 2  # bidirectional даёт удвоенный размер
        for out_features in fc_layers:
            fc_modules.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_features = out_features
        self.fc = nn.Sequential(*fc_modules)
        self.classifier = nn.Linear(in_features, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        # LSTM возвращает (outputs, (h_n, c_n))
        outputs, (h_n, c_n) = self.lstm(x)  # [batch, seq_len, hidden*2]
        # Attention: считаем веса по временной размерности (seq_len)
        # attention(outputs) -> [batch, seq_len, 1]
        att_weights = torch.softmax(self.attention(outputs), dim=1)
        # контекстный вектор: сумма по seq_len
        context = torch.sum(att_weights * outputs, dim=1)  # [batch, hidden*2]
        features = self.fc(context)  # [batch, last_fc_dim]
        return self.classifier(features)  # [batch, 1]


def extract_structured_features(keypoints):
    """Извлечение признаков: для всех пар расстояния, для всех троек углы."""
    features = []
    max_idx = NUM_KEYPOINTS - 1
    
    # Расстояния между парами
    for i, j in ALL_DIST_PAIRS:
        if i < len(keypoints) and j < len(keypoints):
            dist = np.linalg.norm(keypoints[i, :2] - keypoints[j, :2])
        else:
            dist = 0.0
        features.extend([
            i / max_idx,
            j / max_idx,
            dist / 500.0
        ])
    
    # Углы между троицами
    for i, j, k in ALL_ANGLE_TRIPLES:
        angle = 0.0
        if i < len(keypoints) and j < len(keypoints) and k < len(keypoints):
            vec_ij = keypoints[i, :2] - keypoints[j, :2]
            vec_kj = keypoints[k, :2] - keypoints[j, :2]
            norm_ij = np.linalg.norm(vec_ij)
            norm_kj = np.linalg.norm(vec_kj)
            if norm_ij > 1e-6 and norm_kj > 1e-6:
                cosine = np.dot(vec_ij, vec_kj) / (norm_ij * norm_kj)
                angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        features.extend([
            i / max_idx,
            j / max_idx,
            k / max_idx,
            angle / np.pi
        ])
    
    return np.array(features, dtype=np.float32)


class DefecationDetector:
    def __init__(self, lstm_path, dog_detect_model, pose_model, window_size=SEQ_LENGTH, threshold=0.7, smooth=5, progress_callback=None, frame_skip=1):
        self.device = DEVICE
        
        # Загрузка моделей
        self.dog_detect_model = dog_detect_model
        self.pose_model = pose_model
        self.human_detect_model = YOLO("yolov8n.pt")
        self.human_pose_model = YOLO("yolov8s-pose.pt")
        
        self.net = self._load_lstm(lstm_path)
        self.window = deque(maxlen=window_size)
        self.threshold = threshold
        self.smooth = smooth
        self.hist = []
        self.progress_callback = progress_callback  # Callback для обновления прогресса
        self.frame_skip = frame_skip  # Пропускать каждый N-й кадр для ускорения (1 = обрабатывать все кадры)
        
        # Инициализация всех атрибутов состояния
        self.alert = False
        self.defecation_confirmed = False
        self.cleaning_detected = False
        self.prev_dog_feats = None
        self.defecation_point = None
        self.cleaning_min_duration = 2
        self.cleaning_radius = 50
        self.last_defecation_frame = 0
        self.defecation_point_fixed = None
        self.violation_active = False
        self.violation_periods = []  # Список периодов нарушений [(start_frame, end_frame)]
        self.violation_start_frame = None  # Начало текущего периода нарушения
        
        # Цветовые палитры
        self.dog_keypoint_colors = self._generate_color_palette(NUM_KEYPOINTS)
        self.human_keypoint_colors = self._generate_color_palette(17)
        
        # Параметры времени (в кадрах)
        self.fps = 30  # по умолчанию, обновится при обработке видео
        self.defecation_min_duration_frames = 2 * self.fps  # 2 секунды для подтверждения дефекации
        self.cleaning_timeout_frames = 60 * self.fps  # 1 минута ожидания уборки от хозяина
        self.cleaning_min_duration_frames = 5 * self.fps  # 5 секунд для подтверждения уборки
        self.min_defecation_interval_frames = 20 * self.fps  # 20 секунд между дефекациями
        
        # Счетчики кадров
        self.alert_frame_start = None
        self.defecation_frame_fixed = None
        self.cleaning_frame_start = None

    def _generate_color_palette(self, n_colors):
        palette = []
        for hue in np.linspace(0, 179, n_colors):
            color = np.uint8([[[hue, 255, 255]]])
            bgr_color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]
            palette.append(tuple(map(int, bgr_color)))
        return palette

    def _load_lstm(self, path):
        base_feat_len = len(ALL_DIST_PAIRS) * 3 + len(ALL_ANGLE_TRIPLES) * 4
        total_feat_len = base_feat_len * 2
        model = LSTMPoseClassifier(input_size=total_feat_len).to(self.device)
        state = torch.load(path, map_location='cpu')
        model.load_state_dict(state)
        model.eval()
        return model

    def _get_detections(self, frame, target_class="dog"):
        """Детекция объектов с пониженным порогом для собак"""
        min_conf = 0.3 if target_class == "dog" else 0.5
        model = self.human_detect_model if target_class == "person" else self.dog_detect_model
        results = model(frame, verbose=False)[0]
        detections = []
        
        for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(), 
                                  results.boxes.cls.cpu().numpy(), 
                                  results.boxes.conf.cpu().numpy()):
            if conf < min_conf:
                continue
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            
            # Фильтр по размеру
            if target_class == "person":
                if w < 100 or h < 100:
                    continue
            else:
                if w < 30 or h < 30:
                    continue
            
            if target_class == "person":
                if int(cls) == 0:
                    detections.append(([x1, y1, w, h], float(conf), "person"))
            else:
                if results.names[int(cls)] == target_class:
                    detections.append(([x1, y1, w, h], float(conf), target_class))
        
        return detections

    def _calculate_defecation_point(self, dog_kps):
        """Расчет точки дефекации"""
        # Если точка уже зафиксирована - не меняем её
        if self.defecation_point_fixed:
            return self.defecation_point_fixed
        
        # Приоритет: точка 11 (анус)
        if len(dog_kps) > 11 and dog_kps[11][0] > 0 and dog_kps[11][1] > 0:
            return (int(dog_kps[11][0]), int(dog_kps[11][1]))
        return None

    def _is_hand_near_defecation_point(self, human_kps, defecation_point):
        """Проверяет расстояние от рук до точки дефекации"""
        if defecation_point is None or human_kps is None:
            return False
        
        # Индексы запястей: 9 (левое), 10 (правое)
        left_wrist = None
        right_wrist = None
        
        if len(human_kps) > 9 and human_kps[9][0] > 0:
            left_wrist = human_kps[9]
        if len(human_kps) > 10 and human_kps[10][0] > 0:
            right_wrist = human_kps[10]
        
        for wrist in [left_wrist, right_wrist]:
            if wrist is not None:
                distance = math.sqrt((wrist[0] - defecation_point[0])**2 + 
                                    (wrist[1] - defecation_point[1])**2)
                if distance < self.cleaning_radius:
                    return True
        return False

    def _handle_no_dog_detection(self, vis_frame):
        """Обработка отсутствия собаки без визуального предупреждения"""
        # Размер одного вектора признаков
        feat_len = len(ALL_DIST_PAIRS) * 3 + len(ALL_ANGLE_TRIPLES) * 4
        # Создаем нулевой вектор признаков
        zero = np.zeros(feat_len)
        # Добавляем в окно конкатенацию нулевых признаков
        self.window.append(np.concatenate([zero, zero]))
        # Сбрасываем предыдущие признаки
        self.prev_dog_feats = None
        # Сброс статуса дефекации при потере собаки
        self.alert = False
        self.alert_frame_start = None
        self.defecation_confirmed = False
        return vis_frame

    def process_frame(self, frame, frame_count):
        vis_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # === ОБРАБОТКА ЛЮДЕЙ ===
        human_detections = self._get_detections(vis_frame, "person")
        
        # Визуализация всех детекций людей
        for det in human_detections:
            bbox, conf, cls = det
            x1, y1, w_det, h_det = bbox
            x2, y2 = x1 + w_det, y1 + h_det
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis_frame, f"{cls}: {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Обработка ключевых точек людей
        for det in human_detections:
            bbox, conf, cls = det
            x1, y1, w_det, h_det = bbox
            x2, y2 = x1 + w_det, y1 + h_det
            if w_det >= 10 and h_det >= 10:
                cropped_human = frame[y1:y2, x1:x2]
                if cropped_human.size > 0:
                    pose_results = self.human_pose_model(cropped_human, verbose=False)[0]
                    if pose_results.keypoints is not None and len(pose_results.keypoints) > 0:
                        kps = pose_results.keypoints[0].xy[0].cpu().numpy()
                        kps[:, 0] += x1
                        kps[:, 1] += y1
                        
                        # Отрисовка ключевых точек человека
                        for idx_pt, (px, py) in enumerate(kps):
                            if idx_pt < 17:
                                color = self.human_keypoint_colors[idx_pt]
                                cv2.circle(vis_frame, (int(px), int(py)), 4, color, -1)
        
        # === ОБРАБОТКА СОБАК ===
        dog_detections = self._get_detections(vis_frame, "dog")
        full_kps = None
        
        # Визуализация всех детекций собак
        for det in dog_detections:
            bbox, conf, cls = det
            x1, y1, w_det, h_det = bbox
            x2, y2 = x1 + w_det, y1 + h_det
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(vis_frame, f"{cls}: {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Обработка самой надежной детекции собаки
        if dog_detections:
            # Выбираем детекцию с наибольшей уверенностью
            best_det = max(dog_detections, key=lambda x: x[1])
            bbox, conf, cls = best_det
            x1, y1, w_det, h_det = bbox
            x2, y2 = x1 + w_det, y1 + h_det
            
            # Визуализация выбранной детекции
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(vis_frame, f"DOG: {conf:.2f}", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if w_det >= 10 and h_det >= 10:
                cropped = frame[y1:y2, x1:x2]
                
                # Увеличение мелких изображений
                scale_factor = 1.0
                if cropped.shape[0] < 100 or cropped.shape[1] < 100:
                    scale_factor = 2.0
                    cropped = cv2.resize(cropped, None, fx=scale_factor, fy=scale_factor,
                                       interpolation=cv2.INTER_LINEAR)
                
                if cropped.size > 0:
                    pose_results = self.pose_model(cropped, verbose=False)[0]
                    if pose_results.keypoints is not None and len(pose_results.keypoints) > 0:
                        kps = pose_results.keypoints[0].xy[0].cpu().numpy()
                        # Масштабирование ключевых точек обратно
                        if scale_factor != 1.0:
                            kps /= scale_factor
                        kps[:, 0] += x1
                        kps[:, 1] += y1
                        
                        full_kps = np.zeros((NUM_KEYPOINTS, 2))
                        valid = min(kps.shape[0], NUM_KEYPOINTS)
                        full_kps[:valid] = kps[:valid]
                        
                        # Вычисление точки дефекации (только если не зафиксирована)
                        if not self.defecation_point_fixed:
                            self.defecation_point = self._calculate_defecation_point(full_kps)
                        
                        # Вычисление признаков
                        base_feat = extract_structured_features(full_kps)
                        delta_feat = base_feat - self.prev_dog_feats if self.prev_dog_feats is not None else np.zeros_like(base_feat)
                        self.prev_dog_feats = base_feat.copy()
                        self.window.append(np.concatenate([base_feat, delta_feat]))
                        
                        # Отрисовка ключевых точек собаки
                        for idx_pt, (px, py) in enumerate(full_kps):
                            if idx_pt >= NUM_KEYPOINTS:
                                break
                            color = self.dog_keypoint_colors[idx_pt]
                            cv2.circle(vis_frame, (int(px), int(py)), 6, color, -1)
        
        if full_kps is None:
            vis_frame = self._handle_no_dog_detection(vis_frame)
        
        # === ОБРАБОТКА УБОРКИ ===
        if self.defecation_point_fixed:
            for det in human_detections:
                bbox, conf, cls = det
                x1, y1, w_det, h_det = bbox
                x2, y2 = x1 + w_det, y1 + h_det
                if w_det >= 10 and h_det >= 10:
                    cropped_human = frame[y1:y2, x1:x2]
                    if cropped_human.size > 0:
                        pose_results = self.human_pose_model(cropped_human, verbose=False)[0]
                        if pose_results.keypoints is not None and len(pose_results.keypoints) > 0:
                            kps = pose_results.keypoints[0].xy[0].cpu().numpy()
                            kps[:, 0] += x1
                            kps[:, 1] += y1
                            
                            # Проверка близости рук
                            if self._is_hand_near_defecation_point(kps, self.defecation_point_fixed):
                                # Инициализируем duration_sec перед использованием
                                duration_sec = 0.0
                                if self.cleaning_frame_start is None:
                                    self.cleaning_frame_start = frame_count
                                else:
                                    duration_frames = frame_count - self.cleaning_frame_start
                                    duration_sec = duration_frames / self.fps
                                    if duration_frames >= self.cleaning_min_duration_frames:
                                        self.cleaning_detected = True
                                
                                # Добавляем текст уборки
                                cv2.putText(vis_frame, f"CLEANING: {duration_sec:.1f}s", (x1, y1 - 60),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            else:
                                self.cleaning_frame_start = None
        
        return vis_frame

    def run_video(self, in_path, out_path=None):
        print(f"🔍 Processing video: {in_path}")
        cap = cv2.VideoCapture(in_path)
        
        if not cap.isOpened():
            raise ValueError("Error opening video file")
        
        # Получаем FPS видео
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # значение по умолчанию, если FPS не определен
        
        # Обновляем параметры времени в кадрах с учетом реального FPS
        # frame_count учитывает все кадры видео (включая пропущенные), поэтому
        # расчёт времени автоматически корректен: время = frame_count / fps
        self.defecation_min_duration_frames = int(2 * self.fps)  # 2 секунды для подтверждения дефекации
        self.cleaning_timeout_frames = int(60 * self.fps)  # 1 минута (60 секунд) ожидания уборки от хозяина
        self.cleaning_min_duration_frames = int(5 * self.fps)  # 5 секунд для подтверждения уборки
        self.min_defecation_interval_frames = int(20 * self.fps)  # 20 секунд между дефекациями
        
        # Выводим информацию о параметрах обработки
        print(f"📹 FPS видео: {self.fps:.2f}, пропуск кадров: каждый {self.frame_skip}-й кадр")
        print(f"⏱️  Таймаут уборки: {self.cleaning_timeout_frames} кадров ({60} секунд реального времени)")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                                (int(cap.get(3)), int(cap.get(4)))) if out_path else None
        
        # Отключаем tqdm для Streamlit (прогресс показывается в интерфейсе)
        import sys
        is_streamlit = 'streamlit' in sys.modules
        pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame", disable=is_streamlit)
        frame_count = 0  # счетчик кадров
        
        # Вызываем callback для начального прогресса (0%)
        if self.progress_callback:
            try:
                import sys
                sys.stdout.write(f"[DETECTOR] Вызываю начальный callback: 0% из {total_frames} кадров\n")
                sys.stdout.flush()
                self.progress_callback(0, total_frames, f"Начало обработки видео... Всего кадров: {total_frames}")
                sys.stdout.write(f"[DETECTOR] Начальный callback выполнен успешно\n")
                sys.stdout.flush()
            except Exception as e:
                import sys
                sys.stdout.write(f"[DETECTOR ERROR] Ошибка в начальном callback: {e}\n")
                sys.stdout.flush()
        
        # Инициализация переменной для времени с момента дефекации
        time_since_defecation_frames = 0
        last_vis_frame = None  # Для пропущенных кадров
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Пропускаем кадры для ускорения (обрабатываем каждый N-й кадр)
                if frame_count % self.frame_skip != 0:
                    # Для пропущенных кадров используем последний обработанный кадр
                    if writer and last_vis_frame is not None:
                        writer.write(last_vis_frame)
                    elif writer:
                        writer.write(frame)  # Если нет обработанного кадра, пишем оригинал
                    frame_count += 1
                    pbar.update(1)
                    continue
                
                vis_frame = self.process_frame(frame, frame_count)
                last_vis_frame = vis_frame  # Сохраняем для пропущенных кадров
                
                # Обновляем прогресс каждые 3 кадра для более частых обновлений
                if self.progress_callback and (frame_count % 3 == 0 or frame_count == total_frames - 1):
                    progress_percent = min(int((frame_count + 1) / total_frames * 100), 99)  # Максимум 99% до завершения
                    try:
                        self.progress_callback(progress_percent, total_frames, f"Обработка кадра {frame_count + 1} из {total_frames}")
                    except Exception as e:
                        import sys
                        sys.stdout.write(f"[DETECTOR ERROR] Ошибка в callback прогресса: {e}\n")
                        sys.stdout.flush()
                
                if len(self.window) == self.window.maxlen:
                    seq = torch.tensor(np.array(self.window), dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        prob = torch.sigmoid(self.net(seq)).item()
                    self.hist.append(prob)
                    avg_prob = np.mean(self.hist[-self.smooth:] or [0])
                    
                    # =============================================================
                    # ЛОГИКА ОБРАБОТКИ СОБЫТИЙ (ИСПРАВЛЕННАЯ)
                    # =============================================================
                    # Логика подтверждения дефекации (в кадрах)
                    if avg_prob > self.threshold:
                        # Начало алерта или обновление состояния
                        if self.alert_frame_start is None:
                            self.alert_frame_start = frame_count
                            print(f"🚨 Alert started at frame {frame_count} (prob: {avg_prob:.4f})")
                        
                        # Всегда устанавливаем статус ALERT при превышении порога
                        self.alert = True
                        
                        # Вычисляем длительность алерта в кадрах
                        alert_duration_frames = frame_count - self.alert_frame_start
                        
                        # Условие подтверждения дефекации
                        if (alert_duration_frames >= self.defecation_min_duration_frames and 
                            self.defecation_point is not None):
                            # Проверяем интервал с последней дефекации
                            if (self.last_defecation_frame == 0 or  # Первая дефекация
                                frame_count - self.last_defecation_frame > self.min_defecation_interval_frames):
                                self.defecation_confirmed = True
                                self.defecation_point_fixed = self.defecation_point
                                self.defecation_frame_fixed = frame_count
                                self.last_defecation_frame = frame_count
                                print(f"💩 Defecation CONFIRMED at frame {frame_count}! "
                                      f"Point: {self.defecation_point_fixed}")
                            else:
                                print(f"⚠️ Defecation detected but too soon: "
                                      f"{frame_count - self.last_defecation_frame}/"
                                      f"{self.min_defecation_interval_frames} frames")
                    else:
                        # Сбрасываем алерт при падении вероятности ниже порога
                        if self.alert_frame_start is not None:
                            print(f"✅ Alert ended at frame {frame_count} (prob: {avg_prob:.4f})")
                            self.alert_frame_start = None
                            self.alert = False
                            self.defecation_confirmed = False
                    
                    # Вычисляем время с момента дефекации, если зона зафиксирована
                    # frame_count учитывает все кадры видео (включая пропущенные),
                    # поэтому расчёт времени корректен: реальное время = frame_count / fps
                    if self.defecation_point_fixed:
                        # Разница в кадрах (учитывает все кадры, включая пропущенные)
                        time_since_defecation_frames = frame_count - self.defecation_frame_fixed
                        # Реальное время в секундах с учётом FPS видео
                        time_since_defecation_sec = time_since_defecation_frames / self.fps
                    else:
                        time_since_defecation_frames = 0
                        time_since_defecation_sec = 0.0
                    
                    # Проверка нарушения (в кадрах)
                    # Нарушение фиксируется только если прошла 1 минута (60 секунд) без уборки
                    violation = False
                    if self.defecation_point_fixed:
                        if (time_since_defecation_frames >= self.cleaning_timeout_frames and 
                            not self.cleaning_detected):
                            violation = True
                            self.violation_active = True
                            
                            # Начинаем отслеживание периода нарушения
                            if self.violation_start_frame is None:
                                self.violation_start_frame = frame_count
                                # Выводим информацию с учётом реального времени видео
                                print(f"⛔ VIOLATION started at frame {frame_count} "
                                      f"(через {time_since_defecation_sec:.1f} сек реального времени после дефекации, "
                                      f"FPS: {self.fps:.2f}, пропуск: {self.frame_skip})!")
                        else:
                            # Если нарушение было активно, но теперь уборка началась или зона сброшена
                            if self.violation_active and self.violation_start_frame is not None:
                                # Сохраняем период нарушения
                                self.violation_periods.append((self.violation_start_frame, frame_count))
                                print(f"⛔ VIOLATION period recorded: frames {self.violation_start_frame} - {frame_count}")
                                self.violation_start_frame = None
                    
                    # Сброс зоны при уборке или по истечении времени
                    if (self.cleaning_detected or 
                        (self.defecation_point_fixed and 
                         time_since_defecation_frames > self.cleaning_timeout_frames + 5 * self.fps)):
                        print(f"🔄 Resetting defecation zone at frame {frame_count}")
                        
                        # Если был активный период нарушения, сохраняем его
                        if self.violation_active and self.violation_start_frame is not None:
                            self.violation_periods.append((self.violation_start_frame, frame_count))
                            print(f"⛔ VIOLATION period recorded: frames {self.violation_start_frame} - {frame_count}")
                            self.violation_start_frame = None
                        
                        self.defecation_point_fixed = None
                        self.cleaning_detected = False
                        self.cleaning_frame_start = None
                        violation = False  # Сбросим violation, так как зона сброшена
                        self.violation_active = False
                    
                    # =============================================================
                    # ВИЗУАЛИЗАЦИЯ СТАТУСОВ
                    # =============================================================
                    # Упрощенные статусы для отображения
                    status_lines = []
                    colors = []
                    
                    # Статус собаки
                    status_lines.append(f"DOG: {'ALERT' if self.alert else 'NORMAL'} {avg_prob:.2f}")
                    colors.append((0, 0, 255) if self.alert else (0, 255, 0))
                    
                    # Статус дефекации
                    if self.defecation_confirmed:
                        status_lines.append("DEFECATION: CONFIRMED")
                        colors.append((0, 255, 0))
                    elif self.alert:
                        # Вычисляем оставшееся время до подтверждения
                        if self.alert_frame_start is not None:
                            alert_duration_frames = frame_count - self.alert_frame_start
                            if alert_duration_frames < self.defecation_min_duration_frames:
                                pending_sec = (self.defecation_min_duration_frames - alert_duration_frames) / self.fps
                                status_lines.append(f"DEFECATION: PENDING ({pending_sec:.1f}s)")
                            else:
                                # Длительность набрана, но нет точки дефекации
                                status_lines.append("DEFECATION: PENDING (needs point)")
                        else:
                            status_lines.append("DEFECATION: PENDING")
                        colors.append((0, 165, 255))
                    
                    # Статус уборки
                    # Расчёт оставшегося времени с учётом FPS и пропуска кадров
                    # frame_count учитывает все кадры (включая пропущенные), поэтому
                    # расчёт времени корректен: реальное время = кадры / fps
                    if self.defecation_point_fixed:
                        # Оставшееся время до таймаута (в секундах реального времени)
                        time_left_sec = (self.cleaning_timeout_frames - time_since_defecation_frames) / self.fps
                        if self.cleaning_detected:
                            status_lines.append("CLEANING: CLEANED")
                            colors.append((0, 255, 0))
                        else:
                            # Показываем оставшееся время с точностью до 0.1 секунды
                            status_lines.append(f"CLEANING: WAITING {max(0, time_left_sec):.1f}s")
                            colors.append((0, 0, 255))
                    
                    # Статус нарушения
                    if violation:
                        status_lines.append("VIOLATION: DETECTED!")
                        colors.append((0, 0, 255))
                    
                    # Отображение статусов
                    for i, (text, color) in enumerate(zip(status_lines, colors)):
                        cv2.putText(vis_frame, text, (20, 30 + i*30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Отображение фиксированной зоны КВАДРАТОМ
                    if self.defecation_point_fixed:
                        # Размер квадрата (увеличиваем для лучшей видимости)
                        zone_size = 80
                        x_center, y_center = self.defecation_point_fixed
                        # Координаты квадрата
                        x1 = int(x_center - zone_size/2)
                        y1 = int(y_center - zone_size/2)
                        x2 = int(x_center + zone_size/2)
                        y2 = int(y_center + zone_size/2)
                        # Рисуем красный квадрат с более толстой линией
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        # Заливка полупрозрачным красным
                        overlay = vis_frame.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                        cv2.addWeighted(overlay, 0.3, vis_frame, 0.7, 0, vis_frame)
                        # Подпись зоны
                        cv2.putText(vis_frame, "DEFECATION ZONE", (x1, y1 - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                        # Рисуем крестик в центре зоны
                        cross_size = 20
                        cv2.line(vis_frame, 
                                (int(x_center - cross_size), int(y_center)),
                                (int(x_center + cross_size), int(y_center)),
                                (255, 255, 255), 3)
                        cv2.line(vis_frame,
                                (int(x_center), int(y_center - cross_size)),
                                (int(x_center), int(y_center + cross_size)),
                                (255, 255, 255), 3)
                
                if writer:
                    writer.write(vis_frame)
                else:
                    cv2.imshow('Dog Monitoring System', vis_frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
                
                pbar.update(1)
                frame_count += 1  # увеличиваем счетчик кадров
        
        finally:
            # Сохраняем последний период нарушения, если он активен
            if self.violation_active and self.violation_start_frame is not None:
                self.violation_periods.append((self.violation_start_frame, frame_count))
                print(f"⛔ VIOLATION period recorded (final): frames {self.violation_start_frame} - {frame_count}")
            
            # Вызываем callback для завершения (100%)
            if self.progress_callback:
                try:
                    self.progress_callback(100, total_frames, "Обработка завершена!")
                except Exception as e:
                    print(f"Ошибка в финальном callback: {e}")
            
            cap.release()
            if writer:
                writer.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass  # Игнорируем ошибки в Streamlit
            pbar.close()
        
        # Не выводим print в Streamlit, чтобы не засорять интерфейс
        # print(f"✅ Processing completed. Total frames: {total_frames}")

