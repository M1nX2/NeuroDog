import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
import time

# Настройка страницы
st.set_page_config(
    page_title="NeuroDogs Detector",
    page_icon="🦮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок и описание
st.title("🦮 NeuroDogs Defecation Detector")
st.markdown("""
### Система автоматического контроля за собаками

Загрузите видео для анализа поведения собак и детекции дефекации.

**Поддерживаемые форматы:** MP4, AVI, MOV
""")

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки")
    threshold = st.slider("Порог детекции", 0.0, 1.0, 0.8, 0.05)
    smooth = st.slider("Сглаживание", 1, 10, 5)
    st.markdown("---")
    st.info("💡 **Совет:** Более высокий порог = меньше ложных срабатываний")

# Инициализация моделей (кэширование для избежания повторной загрузки)
@st.cache_resource
def load_models():
    """Загрузка моделей детекции"""
    try:
        from detector import DefecationDetector, dog_detect_model, pose_model, SEQ_LENGTH
        return DefecationDetector, dog_detect_model, pose_model, SEQ_LENGTH
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        st.error(f"❌ Ошибка загрузки моделей: {str(e)}\n\nДетали:\n{error_msg}")
        return None, None, None, None

# Загружаем модели
with st.spinner("🔄 Загрузка моделей... Это может занять некоторое время при первом запуске"):
    DefecationDetector, dog_detect_model, pose_model, SEQ_LENGTH = load_models()

if DefecationDetector is None:
    st.error("❌ Не удалось загрузить модели. Проверьте наличие файлов моделей в папке models/")
    st.stop()

# Загрузка видео с улучшенным интерфейсом
st.markdown("### 📹 Загрузка видео")
uploaded_file = st.file_uploader(
    "Перетащите видео сюда или нажмите для выбора",
    type=["mp4", "avi", "mov"],
    help="Поддерживаются форматы MP4, AVI, MOV"
)

if uploaded_file is not None:
    # Показываем информацию о загруженном файле
    file_size = uploaded_file.size / (1024 * 1024)  # Размер в МБ
    st.info(f"📁 Файл: **{uploaded_file.name}** ({file_size:.2f} МБ)")
    
    # Создаем временный файл для входного видео
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tfile:
        tfile.write(uploaded_file.read())
        input_video_path = tfile.name
    
    # Путь для результата
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_video_path = output_dir / f"processed_{Path(uploaded_file.name).stem}.mp4"
    
    # Кнопка запуска обработки
    if st.button("🚀 Начать обработку", type="primary", use_container_width=True):
        try:
            # Прогресс бар
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Создаем placeholder для видео результата
            video_placeholder = st.empty()
            
            # Инициализация детектора
            status_text.text("🔄 Инициализация детектора...")
            progress_bar.progress(10)
            
            detector = DefecationDetector(
                lstm_path="models/structured_lstm_model_final.pth",
                dog_detect_model=dog_detect_model,
                pose_model=pose_model,
                window_size=SEQ_LENGTH,
                threshold=threshold,
                smooth=smooth
            )
            
            progress_bar.progress(20)
            status_text.text("🎬 Обработка видео... Это может занять некоторое время...")
            
            # Обработка видео
            detector.run_video(str(input_video_path), str(output_video_path))
            
            progress_bar.progress(100)
            status_text.text("✅ Обработка завершена!")
            
            # Успешное сообщение
            st.success("✅ Видео успешно обработано!")
            
            # Показываем результат
            st.markdown("### 📺 Результат обработки")
            
            if output_video_path.exists():
                # Показываем видео
                video_placeholder.video(str(output_video_path))
                
                # Кнопка скачивания
                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="📥 Скачать обработанное видео",
                        data=f.read(),
                        file_name=f"processed_{uploaded_file.name}",
                        mime="video/mp4",
                        use_container_width=True
                    )
                
                # Информация о файле
                output_size = output_video_path.stat().st_size / (1024 * 1024)
                st.caption(f"Размер обработанного файла: {output_size:.2f} МБ")
            else:
                st.error("❌ Обработанное видео не найдено")
            
            # Очистка временных файлов
            try:
                os.unlink(input_video_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"❌ Ошибка при обработке видео: {str(e)}")
            st.exception(e)
            
            # Очистка временных файлов при ошибке
            try:
                os.unlink(input_video_path)
            except:
                pass
else:
    # Инструкции, когда файл не загружен
    st.info("👆 Пожалуйста, загрузите видео для обработки")
    
    # Пример использования
    with st.expander("ℹ️ Как использовать"):
        st.markdown("""
        1. **Загрузите видео** - перетащите файл в область выше или нажмите для выбора
        2. **Настройте параметры** - используйте боковую панель для настройки порога и сглаживания
        3. **Запустите обработку** - нажмите кнопку "Начать обработку"
        4. **Дождитесь результата** - обработка может занять время в зависимости от длины видео
        5. **Скачайте результат** - после обработки вы сможете просмотреть и скачать обработанное видео
        
        **Что делает система:**
        - Детектирует собак на видео
        - Анализирует их позы и поведение
        - Определяет моменты дефекации
        - Отслеживает уборку за собаками
        - Выявляет нарушения (если уборка не произведена вовремя)
        """)

# Футер
st.markdown("---")
st.caption("🦮 NeuroDogs Defecation Detector | Система контроля за собаками")
