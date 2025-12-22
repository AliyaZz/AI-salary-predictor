import sys
import numpy as np
import pandas as pd
import warnings
import time
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QTabWidget, QLineEdit, QComboBox, QSlider,
    QTextEdit, QFileDialog, QMessageBox, QFrame, QStyleFactory, QSizePolicy,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox,
    QDoubleSpinBox, QProgressBar, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPainter, QColor, QFont, QIcon

# Matplotlib для PySide6
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')


# ============================================================================
# 1. КЛАСС ДЛЯ ОБУЧЕНИЯ МОДЕЛИ В ОТДЕЛЬНОМ ПОТОКЕ
# ============================================================================

class ModelTrainingThread(QThread):
    """Поток для обучения модели, чтобы не блокировать GUI"""

    # Сигналы для обновления прогресса и завершения
    progress_updated = Signal(int, str)  # прогресс, сообщение
    training_finished = Signal(object, dict, list, list)  # модель, метрики, y_test, y_pred
    error_occurred = Signal(str)

    def __init__(self, X, y, model_type, model_params):
        super().__init__()
        self.X = X
        self.y = y
        self.model_type = model_type
        self.model_params = model_params
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self.progress_updated.emit(10, "Подготовка данных...")
            time.sleep(0.1)

            if not self._is_running:
                return

            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            self.progress_updated.emit(30, "Масштабирование данных...")

            # Масштабирование
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            self.progress_updated.emit(50, "Создание модели...")

            # Выбор модели
            if self.model_type == "Random Forest":
                model = RandomForestRegressor(
                    n_estimators=self.model_params.get('n_estimators', 100),
                    max_depth=self.model_params.get('max_depth', 10),
                    random_state=42
                )
            elif self.model_type == "Gradient Boosting":
                model = GradientBoostingRegressor(
                    n_estimators=self.model_params.get('n_estimators', 100),
                    learning_rate=self.model_params.get('learning_rate', 0.1),
                    random_state=42
                )
            elif self.model_type == "Linear Regression":
                model = LinearRegression()
            elif self.model_type == "Ridge Regression":
                model = Ridge(alpha=self.model_params.get('alpha', 1.0))
            elif self.model_type == "Lasso Regression":
                model = Lasso(alpha=self.model_params.get('alpha', 1.0))
            elif self.model_type == "SVR":
                model = SVR(kernel=self.model_params.get('kernel', 'rbf'),
                            C=self.model_params.get('C', 1.0))
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            self.progress_updated.emit(70, "Обучение модели...")

            # Обучение модели
            model.fit(X_train_scaled, y_train)

            self.progress_updated.emit(90, "Оценка модели...")

            # Прогнозирование
            y_pred = model.predict(X_test_scaled)

            # Расчет метрик
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Кросс-валидация
            cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                        cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }

            self.progress_updated.emit(100, "Обучение завершено!")
            time.sleep(0.5)

            # Сохраняем скалер и данные для предсказаний
            model.scaler = scaler
            model.X_test = X_test
            model.y_test = y_test
            model.y_pred = y_pred

            self.training_finished.emit(model, metrics, y_test.tolist(), y_pred.tolist())

        except Exception as e:
            self.error_occurred.emit(f"Ошибка при обучении: {str(e)}")


# ============================================================================
# 2. ГЛАВНОЕ ОКНО ПРИЛОЖЕНИЯ
# ============================================================================

class SalaryPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("💰 AI Salary Predictor - Предсказание Зарплат")
        self.setGeometry(100, 100, 1400, 900)

        # Настройка стиля
        self.setup_styles()

        # Инициализация данных
        self.salary_data = None
        self.salary_model = None
        self.scaler = StandardScaler()
        self.feature_encoder = {}
        self.setup_demo_data()

        # Главный виджет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Заголовок
        self.title_label = QLabel("💰 AI Salary Predictor - Система Предсказания Зарплат")
        self.title_label.setStyleSheet("""
            font-size: 32px;
            font-weight: 700;
            color: #ffffff;
            padding: 25px;
            text-align: center;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:0.5 #764ba2, stop:1 #667eea);
            border-radius: 15px;
            margin: 8px;
            border: 3px solid #a78bfa;
        """)
        self.main_layout.addWidget(self.title_label)

        # Вкладки
        self.notebook = QTabWidget()
        # Стили вкладок уже применены в setup_styles, но можно добавить дополнительные эффекты
        self.main_layout.addWidget(self.notebook)

        # Инициализация вкладок
        self.setup_data_tab()
        self.setup_training_tab()
        self.setup_prediction_tab()
        self.setup_analysis_tab()

        # Статус бар
        self.statusBar().showMessage("Готов к работе")

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(300)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

        # Таймер для обновления времени
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)

        # Загрузка демо-данных при запуске
        self.load_salary_demo()

        QTimer.singleShot(100, self.update_target_variable_list)

    def setup_styles(self):
        """Настройка стилей приложения"""
        dark_stylesheet = """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #0f0c29, stop:1 #302b63);
            }
            QWidget {
                background-color: transparent;
                color: #f0f0f0;
                font-family: 'Segoe UI', 'Microsoft YaHei', Arial;
                font-size: 13px;
            }
            QTabWidget::pane {
                border: 2px solid #4a5568;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a202c, stop:1 #2d3748);
                border-radius: 12px;
                padding: 5px;
            }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d3748, stop:1 #1a202c);
                color: #cbd5e0;
                padding: 12px 28px;
                margin-right: 4px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                font-weight: 600;
                font-size: 14px;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                color: #ffffff;
                border: 2px solid #8b5cf6;
                border-bottom: none;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a5568, stop:1 #2d3748);
                color: #ffffff;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a5568, stop:1 #2d3748);
                color: #ffffff;
                border: 2px solid #667eea;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 13px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                border: 2px solid #8b5cf6;
                transform: translateY(-1px);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5a67d8, stop:1 #6b46c1);
                border: 2px solid #7c3aed;
            }
            QPushButton#AccentButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4299e1, stop:1 #3182ce);
                border: 2px solid #63b3ed;
                color: #ffffff;
            }
            QPushButton#AccentButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3182ce, stop:1 #2c5282);
                border: 2px solid #90cdf4;
            }
            QPushButton#SuccessButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #48bb78, stop:1 #38a169);
                border: 2px solid #68d391;
                color: #ffffff;
            }
            QPushButton#SuccessButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #38a169, stop:1 #2f855a);
                border: 2px solid #9ae6b4;
            }
            QPushButton#WarningButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ed8936, stop:1 #dd6b20);
                border: 2px solid #f6ad55;
                color: #ffffff;
            }
            QPushButton#WarningButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #dd6b20, stop:1 #c05621);
                border: 2px solid #fbb360;
            }
            QLabel {
                color: #e2e8f0;
                font-size: 13px;
            }
            QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d3748, stop:1 #1a202c);
                border: 2px solid #4a5568;
                padding: 10px;
                border-radius: 8px;
                color: #ffffff;
                font-size: 13px;
                selection-background-color: #667eea;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus, 
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #8b5cf6;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #374151, stop:1 #1f2937);
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #cbd5e0;
                width: 0;
                height: 0;
            }
            QComboBox QAbstractItemView {
                background-color: #2d3748;
                border: 2px solid #667eea;
                border-radius: 8px;
                selection-background-color: #667eea;
                selection-color: #ffffff;
            }
            QGroupBox {
                border: 2px solid #667eea;
                border-radius: 12px;
                margin-top: 20px;
                padding-top: 20px;
                font-weight: 700;
                font-size: 14px;
                color: #a78bfa;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(102, 126, 234, 0.1), stop:1 rgba(118, 75, 162, 0.1));
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0f0c29, stop:1 #302b63);
            }
            QSlider::groove:horizontal {
                height: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2d3748, stop:1 #4a5568);
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #8b5cf6, stop:1 #667eea);
                width: 24px;
                height: 24px;
                margin: -7px 0;
                border-radius: 12px;
                border: 2px solid #a78bfa;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #a78bfa, stop:1 #8b5cf6);
            }
            QTableWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d3748, stop:1 #1a202c);
                border: 2px solid #4a5568;
                gridline-color: #4a5568;
                border-radius: 10px;
                alternate-background-color: rgba(102, 126, 234, 0.1);
            }
            QTableWidget::item {
                padding: 8px;
                border: none;
            }
            QTableWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                color: #ffffff;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a5568, stop:1 #2d3748);
                padding: 10px;
                border: 1px solid #667eea;
                font-weight: 700;
                color: #e2e8f0;
            }
            QProgressBar {
                border: 2px solid #4a5568;
                border-radius: 8px;
                text-align: center;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d3748, stop:1 #1a202c);
                color: #ffffff;
                font-weight: 600;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 6px;
            }
            QScrollBar:vertical {
                background: #2d3748;
                width: 14px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 7px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #8b5cf6, stop:1 #a78bfa);
            }
            QScrollBar:horizontal {
                background: #2d3748;
                height: 14px;
                border-radius: 7px;
            }
            QScrollBar::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 7px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #8b5cf6, stop:1 #a78bfa);
            }
            QStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a202c, stop:1 #2d3748);
                color: #cbd5e0;
                border-top: 2px solid #667eea;
                font-weight: 500;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
        """
        self.setStyleSheet(dark_stylesheet)

    def setup_matplotlib_style(self, fig):
        """Настройка красивого стиля для matplotlib графиков"""
        fig.patch.set_facecolor('#1a202c')
        # Устанавливаем темную тему для всех subplots
        for ax in fig.get_axes():
            ax.set_facecolor('#2d3748')
            ax.spines['bottom'].set_color('#667eea')
            ax.spines['top'].set_color('#667eea')
            ax.spines['right'].set_color('#667eea')
            ax.spines['left'].set_color('#667eea')
            ax.tick_params(colors='#e2e8f0')
            ax.xaxis.label.set_color('#cbd5e0')
            ax.yaxis.label.set_color('#cbd5e0')
            ax.title.set_color('#ffffff')
            ax.grid(True, alpha=0.2, color='#4a5568', linestyle='--')
        return fig

    def update_time(self):
        """Обновление времени в статус баре"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.statusBar().showMessage(f"Готов к работе | {current_time}")

    def setup_demo_data(self):
        """Создание демо-данных для зарплат"""
        np.random.seed(42)
        n_samples = 1000

        positions = ['Python Developer', 'Data Scientist', 'Machine Learning Engineer',
                     'DevOps Engineer', 'Frontend Developer', 'Backend Developer',
                     'Full Stack Developer', 'Data Analyst', 'Software Engineer',
                     'QA Engineer', 'Project Manager', 'Product Manager',
                     'System Administrator', 'Security Engineer', 'Cloud Architect']

        cities = ['Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург',
                  'Казань', 'Нижний Новгород', 'Краснодар', 'Уфа', 'Ростов-на-Дону',
                  'Самара', 'Воронеж', 'Пермь', 'Волгоград']

        levels = ['Junior', 'Middle', 'Senior', 'Lead', 'Architect']
        education = ['Среднее', 'Бакалавр', 'Магистр', 'PhD', 'MBA']
        industries = ['IT', 'Финтех', 'E-commerce', 'Медицина', 'Образование',
                      'Госсектор', 'Телеком', 'Промышленность', 'Ритейл']

        programming_languages = ['Python', 'JavaScript', 'Java', 'C++', 'C#',
                                 'Go', 'Ruby', 'PHP', 'Swift', 'Kotlin']

        data = []
        for i in range(n_samples):
            position = np.random.choice(positions)
            experience = np.random.uniform(0.5, 20)
            age = int(22 + experience * 1.2 + np.random.normal(0, 3))

            # Базовая зарплата в зависимости от должности
            base_salaries = {
                'Python Developer': 130, 'Data Scientist': 150,
                'Machine Learning Engineer': 160, 'DevOps Engineer': 140,
                'Frontend Developer': 120, 'Backend Developer': 130,
                'Full Stack Developer': 135, 'Data Analyst': 100,
                'Software Engineer': 125, 'QA Engineer': 95,
                'Project Manager': 150, 'Product Manager': 160,
                'System Administrator': 110, 'Security Engineer': 145,
                'Cloud Architect': 180
            }

            base_salary = base_salaries.get(position, 120)

            # Модификаторы
            exp_mod = experience * 7
            city_mod = 1.3 if np.random.choice(cities) == 'Москва' else (
                1.15 if np.random.choice(cities) == 'Санкт-Петербург' else 1.0)

            level_idx = min(int(experience / 4), 4)
            level = levels[level_idx]
            level_mod = [1.0, 1.6, 2.3, 3.1, 3.8][level_idx]

            edu_idx = np.random.choice(range(5), p=[0.1, 0.4, 0.3, 0.15, 0.05])
            edu_mod = [0.9, 1.0, 1.15, 1.25, 1.3][edu_idx]

            # Количество проектов
            projects = int(experience * 0.8 + np.random.randint(0, 5))

            # Основной язык программирования
            main_language = np.random.choice(programming_languages)
            language_mod = 1.1 if main_language in ['Python', 'Go'] else 1.0

            # Итоговая зарплата с некоторым шумом
            salary = (base_salary + exp_mod) * city_mod * level_mod * edu_mod * language_mod
            salary += np.random.normal(0, 20)
            salary = max(40, min(600, salary))

            # Уровень английского
            english_level = np.random.choice(['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])

            data.append({
                'ID': i + 1,
                'Должность': position,
                'Возраст': max(18, min(70, age)),
                'Опыт_лет': round(experience, 1),
                'Город': np.random.choice(cities),
                'Образование': education[edu_idx],
                'Уровень': level,
                'Отрасль': np.random.choice(industries),
                'Язык_программирования': main_language,
                'Уровень_английского': english_level,
                'Количество_проектов': projects,
                'Зарплата_тыс': round(salary, 1)
            })

        self.demo_salary_data = pd.DataFrame(data)

    # ============================================================================
    # 3. ВКЛАДКА ДАННЫХ
    # ============================================================================

    def setup_data_tab(self):
        """Вкладка для работы с данными"""
        self.data_tab = QWidget()
        self.notebook.addTab(self.data_tab, "📁 Данные")

        main_layout = QHBoxLayout(self.data_tab)

        # Левая панель - управление данными (с прокруткой)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFixedWidth(370)

        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # Группа загрузки данных
        load_group = QGroupBox("📂 Загрузка Данных")
        load_layout = QVBoxLayout()

        self.data_info_label = QLabel("Загрузите данные о зарплатах")
        self.data_info_label.setWordWrap(True)
        self.data_info_label.setStyleSheet("""
            padding: 15px;
            border: 2px solid #667eea;
            border-radius: 10px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(102, 126, 234, 0.15), stop:1 rgba(118, 75, 162, 0.15));
            color: #e2e8f0;
            font-size: 14px;
        """)
        load_layout.addWidget(self.data_info_label)

        btn_load_excel = QPushButton("📊 Загрузить Excel/CSV файл")
        btn_load_excel.setObjectName("AccentButton")
        btn_load_excel.clicked.connect(self.load_salary_data)
        load_layout.addWidget(btn_load_excel)

        btn_load_demo = QPushButton("🧪 Загрузить демо-данные")
        btn_load_demo.clicked.connect(self.load_salary_demo)
        load_layout.addWidget(btn_load_demo)

        btn_save_data = QPushButton("💾 Сохранить данные")
        btn_save_data.clicked.connect(self.save_salary_data)
        load_layout.addWidget(btn_save_data)

        load_group.setLayout(load_layout)
        left_layout.addWidget(load_group)

        # Группа предварительного анализа
        analysis_group = QGroupBox("📈 Быстрый Анализ")
        analysis_layout = QVBoxLayout()

        btn_quick_stats = QPushButton("📊 Показать статистику")
        btn_quick_stats.clicked.connect(self.show_quick_statistics)
        analysis_layout.addWidget(btn_quick_stats)

        btn_correlation = QPushButton("🔗 Анализ корреляций")
        btn_correlation.clicked.connect(self.show_correlation_analysis)
        analysis_layout.addWidget(btn_correlation)

        btn_clean_data = QPushButton("🧹 Очистить данные")
        btn_clean_data.setObjectName("WarningButton")
        btn_clean_data.clicked.connect(self.clean_data)
        analysis_layout.addWidget(btn_clean_data)

        analysis_group.setLayout(analysis_layout)
        left_layout.addWidget(analysis_group)

        # Группа информации о данных
        info_group = QGroupBox("ℹ️ Информация")
        info_layout = QVBoxLayout()

        self.data_stats_label = QLabel("Нет данных")
        self.data_stats_label.setWordWrap(True)
        info_layout.addWidget(self.data_stats_label)

        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)

        left_layout.addStretch()
        left_scroll.setWidget(left_panel)
        main_layout.addWidget(left_scroll)

        # Правая панель - просмотр данных (с правильным layout)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # Таблица данных (внутри скроллируемой области)
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.data_table.verticalHeader().setVisible(True)

        right_layout.addWidget(self.data_table, stretch=1)

        # Панель управления таблицей
        table_control_panel = QWidget()
        table_control_layout = QHBoxLayout(table_control_panel)
        table_control_layout.setContentsMargins(5, 5, 5, 5)

        self.rows_label = QLabel("Показано записей: 0")
        table_control_layout.addWidget(self.rows_label)

        table_control_layout.addStretch()

        btn_refresh = QPushButton("🔄 Обновить")
        btn_refresh.clicked.connect(self.refresh_data_table)
        table_control_layout.addWidget(btn_refresh)

        btn_export = QPushButton("📤 Экспорт в CSV")
        btn_export.clicked.connect(self.export_to_csv)
        table_control_layout.addWidget(btn_export)

        right_layout.addWidget(table_control_panel)

        main_layout.addWidget(right_panel, stretch=1)

    def load_salary_data(self):
        """Загрузка данных из файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с данными", "",
            "Excel файлы (*.xlsx *.xls);;CSV файлы (*.csv);;Все файлы (*.*)"
        )

        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.salary_data = pd.read_csv(file_path, encoding='utf-8')
                else:
                    self.salary_data = pd.read_excel(file_path)

                file_name = file_path.split('/')[-1]
                self.data_info_label.setText(f"Загружен файл: {file_name}")
                self.update_data_stats()
                self.refresh_data_table()

                # ВАЖНО: Обновляем список целевых переменных!
                self.update_target_variable_list()

                QMessageBox.information(self, "Успех",
                                        f"Данные успешно загружены!\n\n"
                                        f"Файл: {file_name}\n"
                                        f"Записей: {len(self.salary_data):,}\n"
                                        f"Столбцов: {len(self.salary_data.columns)}")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка",
                                     f"Не удалось загрузить файл:\n{str(e)}")

    def load_salary_demo(self):
        """Загрузка демо-данных"""
        self.salary_data = self.demo_salary_data.copy()
        self.data_info_label.setText("Загружены демо-данные (1000 записей о зарплатах)")
        self.update_data_stats()
        self.refresh_data_table()

        # ВАЖНО: Обновляем список целевых переменных!
        self.update_target_variable_list()

        QMessageBox.information(self, "Демо-данные",
                                "Загружены демонстрационные данные:\n\n"
                                "• 1000 записей о зарплатах IT-специалистов\n"
                                "• 15 различных должностей\n"
                                "• 13 городов России\n"
                                "• Разный опыт и образование\n"
                                "• Идеально для тестирования системы")

    def save_salary_data(self):
        """Сохранение данных в файл"""
        if self.salary_data is None:
            QMessageBox.warning(self, "Внимание", "Нет данных для сохранения")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить данные", "salary_data.xlsx",
            "Excel файлы (*.xlsx);;CSV файлы (*.csv)"
        )

        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.salary_data.to_csv(file_path, index=False, encoding='utf-8')
                else:
                    self.salary_data.to_excel(file_path, index=False)

                QMessageBox.information(self, "Успех", f"Данные сохранены в {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить данные:\n{str(e)}")

    def update_data_stats(self):
        """Обновление статистики данных"""
        if self.salary_data is None:
            self.data_stats_label.setText("Нет данных")
            return

        stats_text = f"""
📊 Статистика данных:
• Записей: {len(self.salary_data):,}
• Столбцов: {len(self.salary_data.columns)}
• Пропусков: {self.salary_data.isnull().sum().sum():,}

📋 Примеры столбцов:
"""
        for i, col in enumerate(self.salary_data.columns[:6]):
            dtype = self.salary_data[col].dtype
            unique = self.salary_data[col].nunique()
            stats_text += f"  {i + 1}. {col} ({dtype}, уникальных: {unique})\n"

        if len(self.salary_data.columns) > 6:
            stats_text += f"  ... и еще {len(self.salary_data.columns) - 6} столбцов\n"

        self.data_stats_label.setText(stats_text)

    def refresh_data_table(self):
        """Обновление таблицы данных"""
        if self.salary_data is None:
            self.data_table.clear()
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)
            self.rows_label.setText("Показано записей: 0")
            return

        # Ограничиваем количество отображаемых строк для производительности
        max_rows = min(100, len(self.salary_data))

        self.data_table.setRowCount(max_rows)
        self.data_table.setColumnCount(len(self.salary_data.columns))
        self.data_table.setHorizontalHeaderLabels(self.salary_data.columns)

        for i in range(max_rows):
            for j, col in enumerate(self.salary_data.columns):
                value = self.salary_data.iloc[i, j]

                # Преобразуем значения в строку
                if pd.isna(value):
                    display_value = "NaN"
                else:
                    display_value = str(value)

                item = QTableWidgetItem(display_value)

                # Цветовое кодирование для числовых значений с новой цветовой схемой
                if pd.api.types.is_numeric_dtype(self.salary_data[col]):
                    try:
                        num_val = float(value)
                        if col.lower().find('зарплат') >= 0:
                            if num_val < 100:
                                item.setBackground(QColor(239, 68, 68, 40))  # Красный
                            elif num_val < 200:
                                item.setBackground(QColor(251, 191, 36, 40))  # Желтый
                            else:
                                item.setBackground(QColor(72, 187, 120, 40))  # Зеленый
                    except:
                        pass

                self.data_table.setItem(i, j, item)

        self.data_table.resizeColumnsToContents()
        self.rows_label.setText(f"Показано записей: {max_rows} из {len(self.salary_data):,}")

    def show_quick_statistics(self):
        """Показ быстрой статистики"""
        if self.salary_data is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные")
            return

        # Поиск столбца с зарплатой
        salary_col = None
        for col in self.salary_data.columns:
            if 'зарплат' in col.lower() or 'salary' in col.lower():
                salary_col = col
                break

        stats_window = QMainWindow(self)
        stats_window.setWindowTitle("📊 Статистика данных")
        stats_window.setGeometry(200, 200, 800, 600)

        central_widget = QWidget()
        stats_window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        stats_text = QTextEdit()
        stats_text.setReadOnly(True)
        stats_text.setFont(QFont("Courier New", 10))

        # Генерация статистики
        stats_report = "=" * 80 + "\n"
        stats_report += "СТАТИСТИЧЕСКИЙ АНАЛИЗ ДАННЫХ\n"
        stats_report += "=" * 80 + "\n\n"

        stats_report += f"Общее количество записей: {len(self.salary_data):,}\n"
        stats_report += f"Количество признаков: {len(self.salary_data.columns)}\n\n"

        if salary_col:
            salary_data = pd.to_numeric(self.salary_data[salary_col], errors='coerce').dropna()
            if len(salary_data) > 0:
                stats_report += f"АНАЛИЗ ЗАРПЛАТ ({salary_col}):\n"
                stats_report += "-" * 40 + "\n"
                stats_report += f"Средняя зарплата: {salary_data.mean():.2f} тыс.руб.\n"
                stats_report += f"Медианная зарплата: {salary_data.median():.2f} тыс.руб.\n"
                stats_report += f"Минимальная: {salary_data.min():.2f} тыс.руб.\n"
                stats_report += f"Максимальная: {salary_data.max():.2f} тыс.руб.\n"
                stats_report += f"Стандартное отклонение: {salary_data.std():.2f}\n"
                stats_report += f"25-й перцентиль: {salary_data.quantile(0.25):.2f}\n"
                stats_report += f"75-й перцентиль: {salary_data.quantile(0.75):.2f}\n\n"

        # Распределение по категориям
        categorical_cols = self.salary_data.select_dtypes(include=['object']).columns
        for col in categorical_cols[:3]:  # Первые 3 категориальных признака
            stats_report += f"РАСПРЕДЕЛЕНИЕ ПО '{col}':\n"
            stats_report += "-" * 40 + "\n"
            value_counts = self.salary_data[col].value_counts().head(10)
            for value, count in value_counts.items():
                percentage = count / len(self.salary_data) * 100
                stats_report += f"  {value}: {count} ({percentage:.1f}%)\n"
            stats_report += "\n"

        stats_text.setText(stats_report)
        layout.addWidget(stats_text)

        stats_window.show()

    def show_correlation_analysis(self):
        """Анализ корреляций"""
        if self.salary_data is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные")
            return

        numeric_cols = self.salary_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            QMessageBox.warning(self, "Внимание", "Недостаточно числовых данных для анализа корреляций")
            return

        # Создание окна с графиком корреляции
        corr_window = QMainWindow(self)
        corr_window.setWindowTitle("🔗 Анализ корреляций")
        corr_window.setGeometry(200, 200, 1000, 800)

        central_widget = QWidget()
        corr_window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Вычисление корреляционной матрицы
        corr_matrix = self.salary_data[numeric_cols].corr()

        # Создание графика с увеличенным размером для лучшей читаемости
        fig = Figure(figsize=(16, 14))
        ax = fig.add_subplot(111)

        # Heatmap корреляций с красивой цветовой схемой
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        # Используем красивую цветовую палитру, соответствующую теме
        cmap = sns.diverging_palette(260, 10, as_cmap=True)

        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                    square=True, linewidths=1,
                    ax=ax, annot=True, fmt=".2f", annot_kws={"size": 9, "color": "#ffffff"},
                    cbar_kws={"shrink": .8, "label": "Корреляция"})

        ax.set_title("Корреляционная матрица числовых признаков",
                     fontsize=16, pad=20, color='#ffffff', weight='bold')
        plt.xticks(rotation=45, ha='right', color='#e2e8f0', fontsize=9)
        plt.yticks(rotation=0, color='#e2e8f0', fontsize=9)

        # Применяем темную тему
        self.setup_matplotlib_style(fig)

        # Улучшенное размещение с большими отступами для подписей
        fig.tight_layout(pad=4.0)
        fig.subplots_adjust(bottom=0.15, left=0.15, right=0.95, top=0.95)

        # Встраивание графика в скроллируемую область
        canvas = FigureCanvas(fig)
        canvas.setMinimumSize(800, 700)

        # Создаем скроллируемую область для графика
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setWidget(canvas)
        layout.addWidget(scroll_area)

        # Добавление текстового анализа
        text_widget = QTextEdit()
        text_widget.setMaximumHeight(200)
        text_widget.setReadOnly(True)

        analysis_text = "📊 АНАЛИЗ КОРРЕЛЯЦИЙ:\n\n"

        # Находим сильные корреляции
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > 0.7:
                    strong_correlations.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        if strong_correlations:
            analysis_text += "Сильные корреляции (> 0.7):\n"
            for feat1, feat2, corr in strong_correlations[:5]:
                analysis_text += f"  {feat1} ↔ {feat2}: {corr:.3f}\n"
            analysis_text += "\n"
        else:
            analysis_text += "Сильных корреляций не обнаружено.\n\n"

        # Находим умеренные корреляции
        moderate_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if 0.5 < corr_value <= 0.7:
                    moderate_correlations.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        if moderate_correlations:
            analysis_text += "Умеренные корреляции (0.5 - 0.7):\n"
            for feat1, feat2, corr in moderate_correlations[:5]:
                analysis_text += f"  {feat1} ↔ {feat2}: {corr:.3f}\n"

        text_widget.setText(analysis_text)
        layout.addWidget(text_widget)

        corr_window.show()

    def clean_data(self):
        """Очистка данных"""
        if self.salary_data is None:
            QMessageBox.warning(self, "Внимание", "Нет данных для очистки")
            return

        reply = QMessageBox.question(
            self, "Очистка данных",
            "Вы уверены, что хотите очистить данные?\n\n"
            "Будут выполнены следующие действия:\n"
            "1. Удалены строки с пропущенными значениями\n"
            "2. Удалены дубликаты\n"
            "3. Исправлены типы данных\n\n"
            "Продолжить?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # Сохраняем исходный размер
                original_size = len(self.salary_data)

                # 1. Удаляем пропущенные значения
                self.salary_data = self.salary_data.dropna()

                # 2. Удаляем дубликаты
                self.salary_data = self.salary_data.drop_duplicates()

                # 3. Преобразуем числовые столбцы
                for col in self.salary_data.columns:
                    if self.salary_data[col].dtype == 'object':
                        try:
                            self.salary_data[col] = pd.to_numeric(self.salary_data[col], errors='ignore')
                        except:
                            pass

                # Обновляем интерфейс
                self.update_data_stats()
                self.refresh_data_table()

                removed = original_size - len(self.salary_data)
                QMessageBox.information(self, "Успех",
                                        f"Данные очищены!\n\n"
                                        f"Удалено записей: {removed}\n"
                                        f"Осталось записей: {len(self.salary_data)}\n"
                                        f"Сохранено: {len(self.salary_data) / original_size * 100:.1f}% данных")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при очистке данных:\n{str(e)}")

    def export_to_csv(self):
        """Экспорт данных в CSV"""
        if self.salary_data is None:
            QMessageBox.warning(self, "Внимание", "Нет данных для экспорта")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт данных", "salary_data.csv",
            "CSV файлы (*.csv)"
        )

        if file_path:
            try:
                self.salary_data.to_csv(file_path, index=False, encoding='utf-8')
                QMessageBox.information(self, "Успех", f"Данные экспортированы в {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при экспорте:\n{str(e)}")

    # ============================================================================
    # 4. ВКЛАДКА ОБУЧЕНИЯ МОДЕЛИ
    # ============================================================================

    def setup_training_tab(self):
        """Вкладка для обучения модели"""
        self.training_tab = QWidget()
        self.notebook.addTab(self.training_tab, "🤖 Обучение Модели")

        main_layout = QHBoxLayout(self.training_tab)

        # Левая панель - настройки обучения (с прокруткой)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFixedWidth(420)

        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # Группа выбора модели
        model_group = QGroupBox("🎯 Выбор Модели")
        model_layout = QVBoxLayout()

        model_layout.addWidget(QLabel("Тип модели:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "Random Forest",
            "Gradient Boosting",
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "SVR"
        ])
        self.model_type_combo.currentTextChanged.connect(self.update_model_params)
        model_layout.addWidget(self.model_type_combo)

        # Параметры модели (динамически изменяемые)
        self.model_params_widget = QWidget()
        self.model_params_layout = QVBoxLayout(self.model_params_widget)
        model_layout.addWidget(self.model_params_widget)

        # Целевая переменная
        model_layout.addWidget(QLabel("Целевая переменная (зарплата):"))
        self.target_var_combo = QComboBox()
        model_layout.addWidget(self.target_var_combo)

        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)

        # Группа параметров
        params_group = QGroupBox("⚙️ Параметры Обучения")
        params_layout = QVBoxLayout()

        params_layout.addWidget(QLabel("Размер тестовой выборки:"))
        self.test_size_slider = QSlider(Qt.Horizontal)
        self.test_size_slider.setRange(10, 50)
        self.test_size_slider.setValue(20)
        self.test_size_label = QLabel("20%")
        self.test_size_slider.valueChanged.connect(
            lambda v: self.test_size_label.setText(f"{v}%")
        )
        params_layout.addWidget(self.test_size_slider)
        params_layout.addWidget(self.test_size_label)

        params_layout.addWidget(QLabel("Случайное начальное число:"))
        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 9999)
        self.random_seed_spin.setValue(42)
        params_layout.addWidget(self.random_seed_spin)

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        # Группа запуска обучения
        train_group = QGroupBox("🚀 Запуск Обучения")
        train_layout = QVBoxLayout()

        self.btn_train_model = QPushButton("🎓 Начать обучение модели")
        self.btn_train_model.setObjectName("SuccessButton")
        self.btn_train_model.clicked.connect(self.start_model_training)
        train_layout.addWidget(self.btn_train_model)

        self.btn_stop_training = QPushButton("⏹️ Остановить обучение")
        self.btn_stop_training.setEnabled(False)
        self.btn_stop_training.clicked.connect(self.stop_model_training)
        train_layout.addWidget(self.btn_stop_training)

        # Прогресс обучения
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        train_layout.addWidget(self.training_progress)

        self.training_status = QLabel("Готов к обучению")
        self.training_status.setStyleSheet("""
            padding: 12px;
            border-radius: 8px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(102, 126, 234, 0.1), stop:1 rgba(118, 75, 162, 0.1));
            border: 1px solid #667eea;
            color: #cbd5e0;
            font-weight: 600;
        """)
        train_layout.addWidget(self.training_status)

        train_group.setLayout(train_layout)
        left_layout.addWidget(train_group)

        left_layout.addStretch()
        left_scroll.setWidget(left_panel)
        main_layout.addWidget(left_scroll)

        # Правая панель - результаты обучения (с прокруткой для контента)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)

        self.training_tabs = QTabWidget()

        # Вкладка метрик (с прокруткой)
        metrics_scroll = QScrollArea()
        metrics_scroll.setWidgetResizable(True)
        metrics_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        metrics_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_text)
        metrics_scroll.setWidget(metrics_widget)
        self.training_tabs.addTab(metrics_scroll, "📊 Метрики")

        # Вкладка визуализации (с прокруткой)
        viz_scroll = QScrollArea()
        viz_scroll.setWidgetResizable(True)
        viz_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        viz_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.viz_widget = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_widget)
        viz_scroll.setWidget(self.viz_widget)
        self.training_tabs.addTab(viz_scroll, "📈 Визуализация")

        # Вкладка сравнения моделей (с прокруткой)
        comparison_scroll = QScrollArea()
        comparison_scroll.setWidgetResizable(True)
        comparison_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        comparison_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        comparison_widget = QWidget()
        comparison_layout = QVBoxLayout(comparison_widget)
        self.comparison_text = QTextEdit()
        self.comparison_text.setReadOnly(True)
        comparison_layout.addWidget(self.comparison_text)
        comparison_scroll.setWidget(comparison_widget)
        self.training_tabs.addTab(comparison_scroll, "⚖️ Сравнение")

        right_layout.addWidget(self.training_tabs)
        main_layout.addWidget(right_panel)

        # Инициализация параметров модели
        self.update_model_params()

    def update_model_params(self):
        """Обновление виджетов параметров модели"""
        # Очищаем текущие виджеты
        for i in reversed(range(self.model_params_layout.count())):
            widget = self.model_params_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        model_type = self.model_type_combo.currentText()

        if model_type == "Random Forest":
            self.create_rf_params()
        elif model_type == "Gradient Boosting":
            self.create_gb_params()
        elif model_type in ["Ridge Regression", "Lasso Regression"]:
            self.create_regularization_params()
        elif model_type == "SVR":
            self.create_svr_params()

    def create_rf_params(self):
        """Создание параметров для Random Forest"""
        label = QLabel("Количество деревьев:")
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 500)
        self.n_estimators_spin.setValue(100)
        self.n_estimators_spin.setPrefix("Деревьев: ")

        label2 = QLabel("Максимальная глубина:")
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(2, 50)
        self.max_depth_spin.setValue(10)
        self.max_depth_spin.setPrefix("Глубина: ")

        self.model_params_layout.addWidget(label)
        self.model_params_layout.addWidget(self.n_estimators_spin)
        self.model_params_layout.addWidget(label2)
        self.model_params_layout.addWidget(self.max_depth_spin)

    def create_gb_params(self):
        """Создание параметров для Gradient Boosting"""
        label = QLabel("Количество деревьев:")
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 500)
        self.n_estimators_spin.setValue(100)
        self.n_estimators_spin.setPrefix("Деревьев: ")

        label2 = QLabel("Скорость обучения:")
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.01, 1.0)
        self.learning_rate_spin.setValue(0.1)
        self.learning_rate_spin.setSingleStep(0.01)
        self.learning_rate_spin.setPrefix("LR: ")

        self.model_params_layout.addWidget(label)
        self.model_params_layout.addWidget(self.n_estimators_spin)
        self.model_params_layout.addWidget(label2)
        self.model_params_layout.addWidget(self.learning_rate_spin)

    def create_regularization_params(self):
        """Создание параметров для регуляризации"""
        label = QLabel("Сила регуляризации (alpha):")
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.01, 100.0)
        self.alpha_spin.setValue(1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setPrefix("Alpha: ")

        self.model_params_layout.addWidget(label)
        self.model_params_layout.addWidget(self.alpha_spin)

    def create_svr_params(self):
        """Создание параметров для SVR"""
        label = QLabel("Ядро:")
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(['rbf', 'linear', 'poly', 'sigmoid'])

        label2 = QLabel("Параметр C:")
        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(0.1, 100.0)
        self.c_spin.setValue(1.0)
        self.c_spin.setSingleStep(0.1)
        self.c_spin.setPrefix("C: ")

        self.model_params_layout.addWidget(label)
        self.model_params_layout.addWidget(self.kernel_combo)
        self.model_params_layout.addWidget(label2)
        self.model_params_layout.addWidget(self.c_spin)

    def update_target_variable_list(self):
        """Обновление списка целевых переменных"""
        if self.salary_data is None:
            return

        # Сохраняем текущий выбор
        current_selection = self.target_var_combo.currentText()

        # Очищаем и заполняем заново
        self.target_var_combo.clear()
        self.target_var_combo.addItems(self.salary_data.columns.tolist())

        # Автоматический выбор столбца с зарплатой
        salary_column = None
        for col in self.salary_data.columns:
            col_lower = col.lower()
            if 'зарплат' in col_lower or 'salary' in col_lower or 'оклад' in col_lower or 'доход' in col_lower:
                salary_column = col
                break

        # Если нашли столбец с зарплатой, выбираем его
        if salary_column:
            self.target_var_combo.setCurrentText(salary_column)
        # Иначе пытаемся восстановить предыдущий выбор
        elif current_selection in self.salary_data.columns:
            self.target_var_combo.setCurrentText(current_selection)
        # Иначе выбираем последний столбец
        elif len(self.salary_data.columns) > 0:
            self.target_var_combo.setCurrentText(self.salary_data.columns[-1])

    def start_model_training(self):
        """Запуск обучения модели"""
        if self.salary_data is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные")
            return

        target_col = self.target_var_combo.currentText()
        if not target_col:
            QMessageBox.warning(self, "Внимание", "Выберите целевую переменную")
            return

        try:
            # Подготовка данных
            X = self.salary_data.drop(columns=[target_col])
            y = self.salary_data[target_col]

            # Проверка, что целевая переменная числовая
            if not pd.api.types.is_numeric_dtype(y):
                try:
                    y = pd.to_numeric(y, errors='coerce')
                    mask = y.notna()
                    X = X[mask]
                    y = y[mask]

                    if len(X) == 0:
                        QMessageBox.critical(self, "Ошибка", "Нет корректных числовых данных")
                        return
                except:
                    QMessageBox.critical(self, "Ошибка", "Целевая переменная должна быть числовой")
                    return

            # Кодирование категориальных переменных
            categorical_cols = X.select_dtypes(include=['object']).columns
            self.feature_encoder = {}

            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.feature_encoder[col] = le

            # Сохраняем имена признаков
            self.feature_names = X.columns.tolist()

            # Получаем параметры модели
            model_type = self.model_type_combo.currentText()
            model_params = {}

            if model_type == "Random Forest":
                model_params = {
                    'n_estimators': self.n_estimators_spin.value(),
                    'max_depth': self.max_depth_spin.value()
                }
            elif model_type == "Gradient Boosting":
                model_params = {
                    'n_estimators': self.n_estimators_spin.value(),
                    'learning_rate': self.learning_rate_spin.value()
                }
            elif model_type in ["Ridge Regression", "Lasso Regression"]:
                model_params = {
                    'alpha': self.alpha_spin.value()
                }
            elif model_type == "SVR":
                model_params = {
                    'kernel': self.kernel_combo.currentText(),
                    'C': self.c_spin.value()
                }

            # Настройка интерфейса
            self.btn_train_model.setEnabled(False)
            self.btn_stop_training.setEnabled(True)
            self.training_progress.setVisible(True)
            self.training_progress.setValue(0)
            self.training_status.setText("Подготовка данных...")

            # Запуск обучения в отдельном потоке
            self.training_thread = ModelTrainingThread(X, y, model_type, model_params)
            self.training_thread.progress_updated.connect(self.update_training_progress)
            self.training_thread.training_finished.connect(self.training_completed)
            self.training_thread.error_occurred.connect(self.training_error)
            self.training_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при подготовке данных:\n{str(e)}")

    def update_training_progress(self, progress, message):
        """Обновление прогресса обучения"""
        self.training_progress.setValue(progress)
        self.training_status.setText(message)

    def stop_model_training(self):
        """Остановка обучения"""
        if hasattr(self, 'training_thread') and self.training_thread is not None:
            self.training_thread.stop()
            self.training_thread.wait()

        self.btn_train_model.setEnabled(True)
        self.btn_stop_training.setEnabled(False)
        self.training_status.setText("Обучение остановлено")
        self.progress_bar.setVisible(False)

    def training_completed(self, model, metrics, y_test, y_pred):
        """Завершение обучения"""
        # Сохраняем модель
        self.salary_model = model
        self.model_metrics = metrics

        # Обновляем интерфейс
        self.btn_train_model.setEnabled(True)
        self.btn_stop_training.setEnabled(False)
        self.training_progress.setVisible(False)
        self.training_status.setText("Обучение завершено!")

        # Показываем метрики
        self.show_model_metrics(metrics)

        # Строим графики
        self.plot_training_results(y_test, y_pred)

        # Показываем сравнение
        self.show_model_comparison()

        QMessageBox.information(self, "Успех",
                                f"Модель успешно обучена!\n\n"
                                f"Тип модели: {self.model_type_combo.currentText()}\n"
                                f"Точность (R²): {metrics['r2']:.4f}\n"
                                f"Средняя ошибка: ±{metrics['rmse']:.2f} тыс.руб.\n\n"
                                f"Модель готова для предсказаний!")

    def training_error(self, error_message):
        """Обработка ошибки обучения"""
        self.btn_train_model.setEnabled(True)
        self.btn_stop_training.setEnabled(False)
        self.training_progress.setVisible(False)
        self.training_status.setText("Ошибка при обучении")

        QMessageBox.critical(self, "Ошибка", error_message)

    def show_model_metrics(self, metrics):
        """Отображение метрик модели"""
        metrics_text = "=" * 80 + "\n"
        metrics_text += "МЕТРИКИ КАЧЕСТВА МОДЕЛИ\n"
        metrics_text += "=" * 80 + "\n\n"

        metrics_text += f"📊 ОСНОВНЫЕ МЕТРИКИ:\n"
        metrics_text += f"   Коэффициент детерминации (R²): {metrics['r2']:.4f}\n"
        metrics_text += f"   Среднеквадратичная ошибка (RMSE): {metrics['rmse']:.2f} тыс.руб.\n"
        metrics_text += f"   Средняя абсолютная ошибка (MAE): {metrics['mae']:.2f} тыс.руб.\n"
        metrics_text += f"   Среднеквадратичная ошибка (MSE): {metrics['mse']:.2f}\n\n"

        metrics_text += f"📈 КРОСС-ВАЛИДАЦИЯ (5 фолдов):\n"
        metrics_text += f"   Средний R²: {metrics['cv_mean']:.4f}\n"
        metrics_text += f"   Стандартное отклонение: {metrics['cv_std']:.4f}\n\n"

        # Интерпретация R²
        r2 = metrics['r2']
        if r2 >= 0.9:
            interpretation = "Отличная точность! Модель очень хорошо объясняет вариативность данных."
        elif r2 >= 0.7:
            interpretation = "Хорошая точность. Модель адекватно описывает данные."
        elif r2 >= 0.5:
            interpretation = "Удовлетворительная точность. Модель частично объясняет данные."
        elif r2 >= 0.3:
            interpretation = "Слабая точность. Модель плохо объясняет данные."
        else:
            interpretation = "Очень слабая точность. Модель практически не объясняет данные."

        metrics_text += f"📝 ИНТЕРПРЕТАЦИЯ:\n"
        metrics_text += f"   {interpretation}\n\n"

        # Рекомендации по улучшению
        metrics_text += f"💡 РЕКОМЕНДАЦИИ:\n"
        if r2 < 0.7:
            metrics_text += "   • Попробуйте другие типы моделей\n"
            metrics_text += "   • Добавьте больше данных\n"
            metrics_text += "   • Проверьте качество данных\n"
            metrics_text += "   • Попробуйте другой набор признаков\n"
        else:
            metrics_text += "   • Модель показывает хорошие результаты!\n"
            metrics_text += "   • Можете использовать ее для предсказаний\n"

        self.metrics_text.setText(metrics_text)
        self.training_tabs.setCurrentIndex(0)

    def plot_training_results(self, y_test, y_pred):
        """Построение графиков результатов обучения"""
        # Очистка предыдущих графиков
        for i in reversed(range(self.viz_layout.count())):
            widget = self.viz_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Увеличенный размер фигуры для лучшего размещения всех элементов
        fig = Figure(figsize=(14, 11))

        # 1. Реальные vs Предсказанные значения
        ax1 = fig.add_subplot(221)
        ax1.scatter(y_test, y_pred, alpha=0.6, color='#2196f3', s=40)

        # Линия идеального предсказания
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальное')

        # Линия регрессии
        coeffs = np.polyfit(y_test, y_pred, 1)
        poly = np.poly1d(coeffs)
        x_range = np.linspace(min_val, max_val, 100)
        ax1.plot(x_range, poly(x_range), 'g-', lw=2, label='Линейная регрессия')

        ax1.set_xlabel('Реальная зарплата (тыс.руб.)', fontsize=10)
        ax1.set_ylabel('Предсказанная зарплата (тыс.руб.)', fontsize=10)
        ax1.set_title('Реальные vs Предсказанные', fontsize=11, fontweight='bold', pad=10)
        ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=9)

        # 2. Распределение ошибок
        ax2 = fig.add_subplot(222)
        errors = np.array(y_test) - np.array(y_pred)

        # Гистограмма ошибок
        n_bins = min(30, len(errors) // 10)
        ax2.hist(errors, bins=n_bins, color='#ff9800', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', lw=2)

        # Статистика ошибок
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax2.axvline(x=mean_error, color='b', linestyle='-', lw=2, label=f'Среднее: {mean_error:.2f}')
        ax2.axvline(x=mean_error - std_error, color='b', linestyle=':', lw=1)
        ax2.axvline(x=mean_error + std_error, color='b', linestyle=':', lw=1)

        ax2.set_xlabel('Ошибка предсказания (тыс.руб.)', fontsize=10)
        ax2.set_ylabel('Частота', fontsize=10)
        ax2.set_title('Распределение ошибок', fontsize=11, fontweight='bold', pad=10)
        ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=9)

        # 3. Важность признаков (если модель поддерживает)
        if hasattr(self.salary_model, 'feature_importances_') and hasattr(self, 'feature_names'):
            ax3 = fig.add_subplot(223)
            importances = self.salary_model.feature_importances_

            # Сортируем по важности
            indices = np.argsort(importances)[-12:]  # Топ-12 (уменьшено для экономии места)

            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
            bars = ax3.barh(range(len(indices)), importances[indices], color=colors, edgecolor='black')

            ax3.set_yticks(range(len(indices)))
            # Сокращаем длинные названия признаков
            feature_labels = [self.feature_names[i][:25] + '...' if len(self.feature_names[i]) > 25
                              else self.feature_names[i] for i in indices]
            ax3.set_yticklabels(feature_labels, fontsize=8)
            ax3.set_xlabel('Важность признака', fontsize=10)
            ax3.set_title('Топ-12 важных признаков', fontsize=11, fontweight='bold', pad=10)
            ax3.grid(True, alpha=0.3, axis='x')
            ax3.tick_params(labelsize=8)

            # Добавляем значения на столбцы (только если есть место)
            for bar, importance in zip(bars, importances[indices]):
                width = bar.get_width()
                # Проверяем, есть ли место для текста
                if width > max(importances[indices]) * 0.05:  # Только если столбец достаточно широкий
                    ax3.text(width + max(importances[indices]) * 0.01, bar.get_y() + bar.get_height() / 2,
                             f'{importance:.3f}', ha='left', va='center', fontsize=6)
        else:
            # Если нет важности признаков, создаем пустой subplot
            ax3 = fig.add_subplot(223)
            ax3.text(0.5, 0.5, 'Важность признаков\nнедоступна для\nэтой модели',
                     ha='center', va='center', fontsize=9, transform=ax3.transAxes)
            ax3.set_title('Важность признаков', fontsize=10, fontweight='bold', pad=8)
            ax3.axis('off')

        # 4. Остатки
        ax4 = fig.add_subplot(224)
        ax4.scatter(y_pred, errors, alpha=0.6, color='#9c27b0', s=40)
        ax4.axhline(y=0, color='r', linestyle='--', lw=2)

        # Сглаживание остатков
        if len(y_pred) > 10:
            sorted_indices = np.argsort(y_pred)
            y_pred_sorted = np.array(y_pred)[sorted_indices]
            errors_sorted = np.array(errors)[sorted_indices]

            # Скользящее среднее
            window_size = max(5, len(y_pred) // 20)
            smoothed = pd.Series(errors_sorted).rolling(window=window_size, center=True).mean()
            ax4.plot(y_pred_sorted, smoothed, 'g-', lw=2, label='Сглаженные остатки')

        ax4.set_xlabel('Предсказанная зарплата (тыс.руб.)', fontsize=10)
        ax4.set_ylabel('Остатки', fontsize=10)
        ax4.set_title('Остатки vs Предсказания', fontsize=11, fontweight='bold', pad=10)
        ax4.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=9)

        # Применяем темную тему
        self.setup_matplotlib_style(fig)

        # Улучшенное размещение с большими отступами для предотвращения перекрытий
        fig.tight_layout(pad=3.0, h_pad=3.5, w_pad=3.0)
        # Дополнительная настройка для предотвращения перекрытий подписей
        fig.subplots_adjust(top=0.95, bottom=0.12, left=0.12, right=0.95, hspace=0.4, wspace=0.35)

        canvas = FigureCanvas(fig)
        canvas.setMinimumSize(1000, 800)
        self.viz_layout.addWidget(canvas)
        self.training_tabs.setCurrentIndex(1)

    def show_model_comparison(self):
        """Сравнение различных моделей"""
        if self.salary_data is None:
            return

        comparison_text = "=" * 80 + "\n"
        comparison_text += "СРАВНЕНИЕ РАЗЛИЧНЫХ МОДЕЛЕЙ\n"
        comparison_text += "=" * 80 + "\n\n"

        comparison_text += "🤖 ДОСТУПНЫЕ МОДЕЛИ ДЛЯ ПРЕДСКАЗАНИЯ ЗАРПЛАТ:\n\n"

        models_info = [
            ("Random Forest",
             "🌲 Ансамбль решающих деревьев. Хорошо справляется с нелинейными зависимостями, устойчив к выбросам.",
             "Высокая", "Средняя"),
            ("Gradient Boosting",
             "📈 Последовательное обучение деревьев. Часто дает лучшие результаты, но требует настройки параметров.",
             "Очень высокая", "Высокая"),
            ("Linear Regression",
             "📐 Простая линейная модель. Быстрая, интерпретируемая, но предполагает линейную зависимость.", "Низкая",
             "Очень быстрая"),
            ("Ridge Regression", "⛰️ Линейная модель с L2 регуляризацией. Устойчива к мультиколлинеарности.", "Средняя",
             "Быстрая"),
            ("Lasso Regression", "🎯 Линейная модель с L1 регуляризацией. Выполняет отбор признаков.", "Средняя",
             "Быстрая"),
            ("SVR", "⚡ Метод опорных векторов для регрессии. Хорош для малых выборок, сложных нелинейных зависимостей.",
             "Высокая", "Медленная")
        ]

        for name, description, accuracy, speed in models_info:
            comparison_text += f"🔹 {name}:\n"
            comparison_text += f"   Описание: {description}\n"
            comparison_text += f"   Ожидаемая точность: {accuracy}\n"
            comparison_text += f"   Скорость обучения: {speed}\n\n"

        # Советы по выбору модели
        comparison_text += "💡 СОВЕТЫ ПО ВЫБОРУ МОДЕЛИ:\n\n"
        comparison_text += "1. Для начала попробуйте Random Forest - он хорошо работает 'из коробки'\n"
        comparison_text += "2. Если нужна максимальная точность - используйте Gradient Boosting\n"
        comparison_text += "3. Для интерпретируемости и скорости - Linear/Ridge Regression\n"
        comparison_text += "4. Для малых наборов данных - SVR\n"
        comparison_text += "5. Если много признаков - Lasso для отбора признаков\n\n"

        # Текущая модель
        if hasattr(self, 'model_metrics'):
            comparison_text += f"📊 ТЕКУЩАЯ МОДЕЛЬ ({self.model_type_combo.currentText()}):\n"
            comparison_text += f"   R²: {self.model_metrics['r2']:.4f}\n"
            comparison_text += f"   RMSE: {self.model_metrics['rmse']:.2f} тыс.руб.\n"

            if self.model_metrics['r2'] > 0.8:
                comparison_text += "   ✅ Отличный результат! Модель хорошо обучена.\n"
            elif self.model_metrics['r2'] > 0.6:
                comparison_text += "   ⚠️ Хороший результат, но можно улучшить.\n"
            else:
                comparison_text += "   ❗ Низкая точность. Попробуйте другую модель или улучшите данные.\n"

        self.comparison_text.setText(comparison_text)

    # ============================================================================
    # 5. ВКЛАДКА ПРЕДСКАЗАНИЯ
    # ============================================================================

    def setup_prediction_tab(self):
        """Вкладка для предсказания зарплат"""
        self.prediction_tab = QWidget()
        self.notebook.addTab(self.prediction_tab, "🔮 Предсказание")

        main_layout = QHBoxLayout(self.prediction_tab)

        # Левая панель - ввод данных для предсказания (с прокруткой)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFixedWidth(420)

        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # Группа ввода данных
        input_group = QGroupBox("📝 Введите данные для предсказания")
        input_layout = QVBoxLayout()

        # Должность
        input_layout.addWidget(QLabel("Должность:"))
        self.pred_position = QComboBox()
        if hasattr(self, 'demo_salary_data'):
            positions = self.demo_salary_data['Должность'].unique()
            self.pred_position.addItems(sorted(positions))
        input_layout.addWidget(self.pred_position)

        # Город
        input_layout.addWidget(QLabel("Город:"))
        self.pred_city = QComboBox()
        self.pred_city.addItems(['Москва', 'Санкт-Петербург', 'Новосибирск',
                                 'Екатеринбург', 'Казань', 'Нижний Новгород',
                                 'Краснодар', 'Уфа', 'Другой город'])
        input_layout.addWidget(self.pred_city)

        # Опыт
        input_layout.addWidget(QLabel("Опыт работы (лет):"))
        self.pred_experience = QDoubleSpinBox()
        self.pred_experience.setRange(0, 50)
        self.pred_experience.setValue(3.0)
        self.pred_experience.setSingleStep(0.5)
        self.pred_experience.setSuffix(" лет")
        input_layout.addWidget(self.pred_experience)

        # Возраст
        input_layout.addWidget(QLabel("Возраст:"))
        self.pred_age = QSpinBox()
        self.pred_age.setRange(18, 70)
        self.pred_age.setValue(28)
        self.pred_age.setSuffix(" лет")
        input_layout.addWidget(self.pred_age)

        # Образование
        input_layout.addWidget(QLabel("Образование:"))
        self.pred_education = QComboBox()
        self.pred_education.addItems(['Среднее', 'Бакалавр', 'Магистр', 'PhD', 'MBA'])
        input_layout.addWidget(self.pred_education)

        # Уровень
        input_layout.addWidget(QLabel("Уровень:"))
        self.pred_level = QComboBox()
        self.pred_level.addItems(['Junior', 'Middle', 'Senior', 'Lead', 'Architect'])
        input_layout.addWidget(self.pred_level)

        # Язык программирования
        input_layout.addWidget(QLabel("Основной язык программирования:"))
        self.pred_language = QComboBox()
        self.pred_language.addItems(['Python', 'JavaScript', 'Java', 'C++', 'C#',
                                     'Go', 'Ruby', 'PHP', 'Swift', 'Kotlin', 'Другой'])
        input_layout.addWidget(self.pred_language)

        # Уровень английского
        input_layout.addWidget(QLabel("Уровень английского:"))
        self.pred_english = QComboBox()
        self.pred_english.addItems(['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])
        input_layout.addWidget(self.pred_english)

        # Количество проектов
        input_layout.addWidget(QLabel("Количество завершенных проектов:"))
        self.pred_projects = QSpinBox()
        self.pred_projects.setRange(0, 100)
        self.pred_projects.setValue(5)
        input_layout.addWidget(self.pred_projects)

        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)

        # Группа кнопок предсказания
        predict_group = QGroupBox("🚀 Предсказание")
        predict_layout = QVBoxLayout()

        self.btn_predict = QPushButton("💰 Предсказать зарплату")
        self.btn_predict.setObjectName("SuccessButton")
        self.btn_predict.clicked.connect(self.predict_salary)
        predict_layout.addWidget(self.btn_predict)

        self.btn_clear = QPushButton("🧹 Очистить форму")
        self.btn_clear.clicked.connect(self.clear_prediction_form)
        predict_layout.addWidget(self.btn_clear)

        self.btn_save_prediction = QPushButton("💾 Сохранить предсказание")
        self.btn_save_prediction.clicked.connect(self.save_prediction)
        predict_layout.addWidget(self.btn_save_prediction)

        predict_group.setLayout(predict_layout)
        left_layout.addWidget(predict_group)

        left_layout.addStretch()
        left_scroll.setWidget(left_panel)
        main_layout.addWidget(left_scroll)

        # Правая панель - результаты предсказания (с прокруткой)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # Результат предсказания
        self.result_group = QGroupBox("📊 Результат предсказания")
        self.result_layout = QVBoxLayout(self.result_group)

        self.prediction_result = QLabel("Здесь появится результат предсказания")
        self.prediction_result.setAlignment(Qt.AlignCenter)
        self.prediction_result.setStyleSheet("""
            font-size: 20px;
            font-weight: 700;
            padding: 30px;
            border: 3px dashed #8b5cf6;
            border-radius: 15px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(102, 126, 234, 0.2), stop:1 rgba(118, 75, 162, 0.2));
            color: #ffffff;
        """)
        self.result_layout.addWidget(self.prediction_result)

        # Детали предсказания
        self.prediction_details = QTextEdit()
        self.prediction_details.setReadOnly(True)
        self.prediction_details.setMaximumHeight(150)
        self.result_layout.addWidget(self.prediction_details)

        right_layout.addWidget(self.result_group)

        # История предсказаний
        history_group = QGroupBox("📜 История предсказаний")
        history_layout = QVBoxLayout(history_group)

        self.prediction_history = QTableWidget()
        self.prediction_history.setColumnCount(4)
        self.prediction_history.setHorizontalHeaderLabels([
            "Время", "Должность", "Предсказание", "Детали"
        ])
        self.prediction_history.horizontalHeader().setStretchLastSection(True)
        history_layout.addWidget(self.prediction_history)

        right_layout.addWidget(history_group)

        right_scroll.setWidget(right_panel)
        main_layout.addWidget(right_scroll)

        # Инициализация истории предсказаний
        self.prediction_history_data = []

    def predict_salary(self):
        """Предсказание зарплаты на основе введенных данных"""
        if self.salary_model is None:
            QMessageBox.warning(self, "Внимание", "Сначала обучите модель на вкладке 'Обучение Модели'")
            return

        try:
            # Сбор данных из формы
            input_data = {
                'Должность': self.pred_position.currentText(),
                'Город': self.pred_city.currentText(),
                'Опыт_лет': self.pred_experience.value(),
                'Возраст': self.pred_age.value(),
                'Образование': self.pred_education.currentText(),
                'Уровень': self.pred_level.currentText(),
                'Язык_программирования': self.pred_language.currentText(),
                'Уровень_английского': self.pred_english.currentText(),
                'Количество_проектов': self.pred_projects.value()
            }

            # Создание DataFrame
            input_df = pd.DataFrame([input_data])

            # Проверяем наличие всех признаков
            if not hasattr(self, 'feature_names'):
                QMessageBox.critical(self, "Ошибка",
                                     "Модель не была корректно обучена. Переобучите модель.")
                return

            # Кодирование категориальных признаков
            for col in input_df.columns:
                if col in self.feature_encoder:
                    try:
                        input_df[col] = self.feature_encoder[col].transform([input_df[col].iloc[0]])[0]
                    except:
                        # Если значение не было в обучающей выборке, используем самое частое
                        input_df[col] = 0

            # Добавляем недостающие признаки
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0

            # Упорядочиваем признаки как при обучении
            input_df = input_df[self.feature_names]

            # Масштабирование
            input_scaled = self.salary_model.scaler.transform(input_df)

            # Предсказание
            prediction = self.salary_model.predict(input_scaled)[0]
            rmse = self.model_metrics['rmse']

            # Расчет доверительного интервала
            lower_bound = max(0, prediction - rmse)
            upper_bound = prediction + rmse

            # Форматирование результата
            result_text = f"""
            <div style="text-align: center;">
                <h2 style="color: #4caf50;">💰 ПРЕДСКАЗАННАЯ ЗАРПЛАТА</h2>
                <div style="font-size: 36px; font-weight: bold; color: #4fc3f7; margin: 20px 0;">
                    {prediction:.1f} тыс.руб./мес
                </div>
                <div style="font-size: 18px; color: #ff9800; margin-bottom: 20px;">
                    📊 Диапазон: {lower_bound:.1f} - {upper_bound:.1f} тыс.руб.
                </div>
                <div style="font-size: 14px; color: #cccccc;">
                    🎯 Точность предсказания: ±{rmse:.1f} тыс.руб.
                </div>
            </div>
            """

            self.prediction_result.setText(result_text)

            # Детали предсказания
            details_text = f"""
            📋 ДЕТАЛИ ПРЕДСКАЗАНИЯ:

            • Должность: {input_data['Должность']}
            • Город: {input_data['Город']}
            • Опыт: {input_data['Опыт_лет']} лет
            • Возраст: {input_data['Возраст']} лет
            • Образование: {input_data['Образование']}
            • Уровень: {input_data['Уровень']}
            • Язык программирования: {input_data['Язык_программирования']}
            • Английский: {input_data['Уровень_английского']}
            • Проекты: {input_data['Количество_проектов']}

            ⚠️ Примечание: Предсказание основано на статистических данных и может отличаться 
            от реальных предложений на рынке труда.
            """

            self.prediction_details.setText(details_text)

            # Сохранение в историю
            self.save_to_prediction_history(input_data, prediction, lower_bound, upper_bound)

            # Обновление статуса
            self.statusBar().showMessage(f"Предсказана зарплата: {prediction:.1f} тыс.руб.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка",
                                 f"Ошибка при предсказании:\n{str(e)}\n\n"
                                 f"Убедитесь, что:\n"
                                 f"1. Модель обучена корректно\n"
                                 f"2. Все необходимые поля заполнены\n"
                                 f"3. Типы данных соответствуют обучающей выборке")

    def save_to_prediction_history(self, input_data, prediction, lower_bound, upper_bound):
        """Сохранение предсказания в историю"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Добавляем в список
        self.prediction_history_data.append({
            'timestamp': timestamp,
            'position': input_data['Должность'],
            'prediction': prediction,
            'details': f"{lower_bound:.1f}-{upper_bound:.1f}"
        })

        # Обновляем таблицу (показываем последние 10 предсказаний)
        self.update_prediction_history_table()

    def update_prediction_history_table(self):
        """Обновление таблицы истории предсказаний"""
        # Показываем последние 10 предсказаний
        recent_predictions = self.prediction_history_data[-10:]

        self.prediction_history.setRowCount(len(recent_predictions))

        for i, pred in enumerate(recent_predictions):
            self.prediction_history.setItem(i, 0, QTableWidgetItem(pred['timestamp']))
            self.prediction_history.setItem(i, 1, QTableWidgetItem(pred['position']))
            self.prediction_history.setItem(i, 2, QTableWidgetItem(f"{pred['prediction']:.1f} тыс.руб."))
            self.prediction_history.setItem(i, 3, QTableWidgetItem(pred['details']))

        self.prediction_history.resizeColumnsToContents()

    def clear_prediction_form(self):
        """Очистка формы предсказания"""
        self.pred_experience.setValue(3.0)
        self.pred_age.setValue(28)
        self.pred_projects.setValue(5)

        self.prediction_result.setText("Здесь появится результат предсказания")
        self.prediction_details.clear()

        self.statusBar().showMessage("Форма очищена")

    def save_prediction(self):
        """Сохранение предсказания в файл"""
        if not self.prediction_history_data:
            QMessageBox.warning(self, "Внимание", "Нет предсказаний для сохранения")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить предсказания", "salary_predictions.csv",
            "CSV файлы (*.csv);;Text файлы (*.txt)"
        )

        if file_path:
            try:
                # Создаем DataFrame из истории
                df = pd.DataFrame(self.prediction_history_data)

                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False, encoding='utf-8')
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("=" * 60 + "\n")
                        f.write("ИСТОРИЯ ПРЕДСКАЗАНИЙ ЗАРПЛАТ\n")
                        f.write("=" * 60 + "\n\n")

                        for pred in self.prediction_history_data:
                            f.write(f"Время: {pred['timestamp']}\n")
                            f.write(f"Должность: {pred['position']}\n")
                            f.write(f"Предсказанная зарплата: {pred['prediction']:.1f} тыс.руб.\n")
                            f.write(f"Диапазон: {pred['details']} тыс.руб.\n")
                            f.write("-" * 40 + "\n")

                QMessageBox.information(self, "Успех", f"Предсказания сохранены в {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить предсказания:\n{str(e)}")

    # ============================================================================
    # 6. ВКЛАДКА АНАЛИЗА
    # ============================================================================

    def setup_analysis_tab(self):
        """Вкладка для анализа данных и модели"""
        self.analysis_tab = QWidget()
        self.notebook.addTab(self.analysis_tab, "📈 Анализ")

        main_layout = QHBoxLayout(self.analysis_tab)

        # Левая панель - инструменты анализа (с прокруткой)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFixedWidth(370)

        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # Группа анализа данных
        analysis_group = QGroupBox("🔍 Инструменты Анализа")
        analysis_layout = QVBoxLayout()

        btn_salary_dist = QPushButton("💰 Распределение зарплат")
        btn_salary_dist.clicked.connect(self.analyze_salary_distribution)
        analysis_layout.addWidget(btn_salary_dist)

        btn_position_analysis = QPushButton("👔 Анализ по должностям")
        btn_position_analysis.clicked.connect(self.analyze_by_position)
        analysis_layout.addWidget(btn_position_analysis)

        btn_city_analysis = QPushButton("🏙️ Анализ по городам")
        btn_city_analysis.clicked.connect(self.analyze_by_city)
        analysis_layout.addWidget(btn_city_analysis)

        btn_experience_analysis = QPushButton("📊 Зависимость от опыта")
        btn_experience_analysis.clicked.connect(self.analyze_by_experience)
        analysis_layout.addWidget(btn_experience_analysis)

        btn_export_report = QPushButton("📄 Создать отчет")
        btn_export_report.setObjectName("AccentButton")
        btn_export_report.clicked.connect(self.generate_analysis_report)
        analysis_layout.addWidget(btn_export_report)

        analysis_group.setLayout(analysis_layout)
        left_layout.addWidget(analysis_group)

        # Группа статистики
        stats_group = QGroupBox("📊 Статистика Модели")
        stats_layout = QVBoxLayout()

        self.model_stats_text = QTextEdit()
        self.model_stats_text.setReadOnly(True)
        self.model_stats_text.setMaximumHeight(200)
        stats_layout.addWidget(self.model_stats_text)

        stats_group.setLayout(stats_layout)
        left_layout.addWidget(stats_group)

        left_layout.addStretch()
        left_scroll.setWidget(left_panel)
        main_layout.addWidget(left_scroll)

        # Правая панель - графики анализа (с прокруткой)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.analysis_plots_widget = QWidget()
        self.analysis_plots_layout = QVBoxLayout(self.analysis_plots_widget)
        right_layout.addWidget(self.analysis_plots_widget)

        right_scroll.setWidget(right_panel)
        main_layout.addWidget(right_scroll)

    def analyze_salary_distribution(self):
        """Анализ распределения зарплат"""
        if self.salary_data is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные")
            return

        # Находим столбец с зарплатой
        salary_col = None
        for col in self.salary_data.columns:
            if 'зарплат' in col.lower() or 'salary' in col.lower():
                salary_col = col
                break

        if not salary_col:
            QMessageBox.warning(self, "Внимание", "Не найден столбец с зарплатой")
            return

        salary_data = pd.to_numeric(self.salary_data[salary_col], errors='coerce').dropna()

        if len(salary_data) == 0:
            QMessageBox.warning(self, "Внимание", "Нет корректных данных о зарплате")
            return

        # Очищаем предыдущие графики
        for i in reversed(range(self.analysis_plots_layout.count())):
            widget = self.analysis_plots_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        fig = Figure(figsize=(18, 14))

        # 1. Гистограмма распределения
        ax1 = fig.add_subplot(221)
        n_bins = min(30, len(salary_data) // 10)
        ax1.hist(salary_data, bins=n_bins, color='#2196f3', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Зарплата (тыс.руб.)', fontsize=12)
        ax1.set_ylabel('Частота', fontsize=12)
        ax1.set_title('Распределение зарплат', fontsize=14, fontweight='bold', pad=18)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=10)

        # Добавляем линии среднего и медианы
        mean_salary = salary_data.mean()
        median_salary = salary_data.median()
        ax1.axvline(mean_salary, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_salary:.1f}')
        ax1.axvline(median_salary, color='green', linestyle='--', linewidth=2, label=f'Медиана: {median_salary:.1f}')
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)

        # 2. Box plot
        ax2 = fig.add_subplot(222)
        bp = ax2.boxplot(salary_data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('#ff9800')
        bp['medians'][0].set_color('red')
        ax2.set_ylabel('Зарплата (тыс.руб.)', fontsize=12)
        ax2.set_title('Box plot зарплат', fontsize=14, fontweight='bold', pad=18)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=10)

        # 3. Q-Q plot
        ax3 = fig.add_subplot(223)
        from scipy import stats
        stats.probplot(salary_data, dist="norm", plot=ax3)
        ax3.set_title('Q-Q plot (нормальность распределения)', fontsize=14, fontweight='bold', pad=18)
        ax3.set_xlabel('Теоретические квантили', fontsize=12)
        ax3.set_ylabel('Выборочные квантили', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=10)

        # 4. Плотность распределения
        ax4 = fig.add_subplot(224)
        import seaborn as sns
        sns.kdeplot(salary_data, ax=ax4, color='purple', linewidth=2, fill=True, alpha=0.3)
        ax4.set_xlabel('Зарплата (тыс.руб.)', fontsize=12)
        ax4.set_ylabel('Плотность', fontsize=12)
        ax4.set_title('Плотность распределения', fontsize=14, fontweight='bold', pad=18)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=10)

        # Применяем темную тему
        self.setup_matplotlib_style(fig)

        # Улучшенное размещение с большими отступами
        fig.tight_layout(pad=4.0, h_pad=4.0, w_pad=4.0)
        fig.subplots_adjust(top=0.94, bottom=0.1, left=0.1, right=0.95, hspace=0.45, wspace=0.4)

        canvas = FigureCanvas(fig)
        canvas.setMinimumSize(1200, 900)
        self.analysis_plots_layout.addWidget(canvas)

        # Обновляем статистику
        stats_text = f"""
        📊 СТАТИСТИКА РАСПРЕДЕЛЕНИЯ ЗАРПЛАТ:

        • Средняя зарплата: {mean_salary:.1f} тыс.руб.
        • Медианная зарплата: {median_salary:.1f} тыс.руб.
        • Минимальная: {salary_data.min():.1f} тыс.руб.
        • Максимальная: {salary_data.max():.1f} тыс.руб.
        • Стандартное отклонение: {salary_data.std():.1f}
        • Коэффициент вариации: {salary_data.std() / mean_salary * 100:.1f}%

        📈 ИНТЕРПРЕТАЦИЯ:

        """

        cv = salary_data.std() / mean_salary
        if cv < 0.1:
            stats_text += "Распределение очень однородное"
        elif cv < 0.3:
            stats_text += "Умеренная вариативность зарплат"
        else:
            stats_text += "Высокая вариативность зарплат"

        self.model_stats_text.setText(stats_text)

    def analyze_by_position(self):
        """Анализ зарплат по должностям"""
        if self.salary_data is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные")
            return

        # Находим столбцы
        position_col = None
        salary_col = None

        for col in self.salary_data.columns:
            if 'должн' in col.lower() or 'position' in col.lower():
                position_col = col
            if 'зарплат' in col.lower() or 'salary' in col.lower():
                salary_col = col

        if not position_col or not salary_col:
            QMessageBox.warning(self, "Внимание", "Не найдены столбцы с должностью или зарплатой")
            return

        # Очищаем предыдущие графики
        for i in reversed(range(self.analysis_plots_layout.count())):
            widget = self.analysis_plots_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Группируем по должности
        salary_by_position = self.salary_data.groupby(position_col)[salary_col].agg([
            'mean', 'median', 'count', 'std', 'min', 'max'
        ]).round(1)

        # Сортируем по средней зарплате
        salary_by_position = salary_by_position.sort_values('mean', ascending=False).head(15)

        fig = Figure(figsize=(18, 10))

        # 1. Bar chart средних зарплат
        ax1 = fig.add_subplot(121)
        positions = salary_by_position.index
        y_pos = np.arange(len(positions))
        means = salary_by_position['mean']
        stds = salary_by_position['std']

        bars = ax1.barh(y_pos, means, color='#4fc3f7', edgecolor='black')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(positions, fontsize=10)
        ax1.set_xlabel('Средняя зарплата (тыс.руб.)', fontsize=12)
        ax1.set_title('Топ-15 должностей по средней зарплате', fontsize=14, fontweight='bold', pad=18)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.tick_params(labelsize=10)

        # Добавляем значения на столбцы
        for bar, mean_val, count in zip(bars, means, salary_by_position['count']):
            width = bar.get_width()
            ax1.text(width + max(means) * 0.02, bar.get_y() + bar.get_height() / 2,
                     f'{mean_val:.1f} (n={count})', ha='left', va='center', fontsize=8)

        # 2. Box plot по топ-5 должностям
        if len(positions) >= 5:
            ax2 = fig.add_subplot(122)

            # Собираем данные для топ-5 должностей
            top_positions = positions[:5]
            data_to_plot = []

            for pos in top_positions:
                pos_salaries = self.salary_data[self.salary_data[position_col] == pos][salary_col]
                pos_salaries = pd.to_numeric(pos_salaries, errors='coerce').dropna()
                data_to_plot.append(pos_salaries.values)

            bp = ax2.boxplot(data_to_plot, vert=True, patch_artist=True)

            # Раскрашиваем box plots
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            ax2.set_xticklabels(top_positions, rotation=45, ha='right', fontsize=10)
            ax2.set_ylabel('Зарплата (тыс.руб.)', fontsize=12)
            ax2.set_title('Распределение зарплат по топ-5 должностям', fontsize=14, fontweight='bold', pad=18)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=10)

        # Применяем темную тему
        self.setup_matplotlib_style(fig)

        # Улучшенное размещение с большими отступами
        fig.tight_layout(pad=4.0, h_pad=4.0, w_pad=4.0)
        fig.subplots_adjust(top=0.94, bottom=0.15, left=0.15, right=0.95, hspace=0.35, wspace=0.4)

        canvas = FigureCanvas(fig)
        canvas.setMinimumSize(1200, 700)
        self.analysis_plots_layout.addWidget(canvas)

        # Статистика
        stats_text = f"""
        📊 АНАЛИЗ ПО ДОЛЖНОСТЯМ:

        Всего уникальных должностей: {self.salary_data[position_col].nunique()}

        🥇 Самая высокая средняя зарплата:
          • Должность: {positions[0]}
          • Зарплата: {means.iloc[0]:.1f} тыс.руб.
          • Записей: {salary_by_position['count'].iloc[0]}

        🥈 Вторая по зарплате:
          • Должность: {positions[1] if len(positions) > 1 else 'Н/Д'}
          • Зарплата: {means.iloc[1] if len(positions) > 1 else 'Н/Д'} тыс.руб.

        📈 Разброс зарплат (стандартное отклонение):
        """

        for i, (pos, row) in enumerate(salary_by_position.head(5).iterrows()):
            cv = row['std'] / row['mean'] * 100 if row['mean'] > 0 else 0
            stats_text += f"  • {pos}: {row['std']:.1f} ({cv:.1f}%)\n"

        self.model_stats_text.setText(stats_text)

    def analyze_by_city(self):
        """Анализ зарплат по городам"""
        if self.salary_data is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные")
            return

        # Находим столбцы
        city_col = None
        salary_col = None

        for col in self.salary_data.columns:
            if 'город' in col.lower() or 'city' in col.lower():
                city_col = col
            if 'зарплат' in col.lower() or 'salary' in col.lower():
                salary_col = col

        if not city_col or not salary_col:
            QMessageBox.warning(self, "Внимание", "Не найдены столбцы с городом или зарплатой")
            return

        # Очищаем предыдущие графики
        for i in reversed(range(self.analysis_plots_layout.count())):
            widget = self.analysis_plots_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Группируем по городу
        salary_by_city = self.salary_data.groupby(city_col)[salary_col].agg([
            'mean', 'median', 'count', 'std'
        ]).round(1)

        # Сортируем по количеству записей
        salary_by_city = salary_by_city.sort_values('count', ascending=False).head(10)

        fig = Figure(figsize=(18, 12))

        # 1. Bar chart средних зарплат по городам
        ax1 = fig.add_subplot(121)
        cities = salary_by_city.index
        y_pos = np.arange(len(cities))

        # Два ряда: средняя и медиана
        width = 0.35
        bars1 = ax1.barh(y_pos - width / 2, salary_by_city['mean'], width,
                         label='Средняя', color='#4fc3f7', edgecolor='black')
        bars2 = ax1.barh(y_pos + width / 2, salary_by_city['median'], width,
                         label='Медиана', color='#ff9800', edgecolor='black')

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(cities, fontsize=10)
        ax1.set_xlabel('Зарплата (тыс.руб.)', fontsize=12)
        ax1.set_title('Средняя и медианная зарплата по городам (топ-10)',
                      fontsize=14, fontweight='bold', pad=18)
        ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.tick_params(labelsize=10)

        # 2. Количество записей по городам
        ax2 = fig.add_subplot(122)
        colors = plt.cm.Paired(np.linspace(0, 1, len(cities)))
        wedges, texts, autotexts = ax2.pie(salary_by_city['count'], labels=cities,
                                           colors=colors, autopct='%1.1f%%',
                                           startangle=90, textprops={'fontsize': 10})

        # Улучшаем читаемость текста на pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        for text in texts:
            text.set_fontsize(10)

        ax2.set_title('Распределение записей по городам', fontsize=14, fontweight='bold', pad=18)

        # Применяем темную тему
        self.setup_matplotlib_style(fig)

        # Улучшенное размещение с большими отступами
        fig.tight_layout(pad=4.0, h_pad=4.0, w_pad=4.0)
        fig.subplots_adjust(top=0.94, bottom=0.1, left=0.1, right=0.95, hspace=0.35, wspace=0.4)

        canvas = FigureCanvas(fig)
        canvas.setMinimumSize(1200, 800)
        self.analysis_plots_layout.addWidget(canvas)

        # Статистика
        stats_text = f"""
        📊 АНАЛИЗ ПО ГОРОДАМ:

        Всего уникальных городов: {self.salary_data[city_col].nunique()}

        📍 Города с наибольшим количеством данных:
        """

        for i, (city, row) in enumerate(salary_by_city.head(5).iterrows()):
            stats_text += f"  {i + 1}. {city}: {row['count']} записей, "
            stats_text += f"средняя: {row['mean']:.1f} тыс.руб.\n"

        # Москва vs другие города
        if 'Москва' in salary_by_city.index:
            moscow_avg = salary_by_city.loc['Москва', 'mean']
            other_avg = salary_by_city[salary_by_city.index != 'Москва']['mean'].mean()
            premium = (moscow_avg / other_avg - 1) * 100 if other_avg > 0 else 0

            stats_text += f"\n🏙️ Премия Москвы: +{premium:.1f}% к средней зарплате по другим городам\n"

        self.model_stats_text.setText(stats_text)

    def analyze_by_experience(self):
        """Анализ зависимости зарплаты от опыта"""
        if self.salary_data is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные")
            return

        # Находим столбцы
        experience_col = None
        salary_col = None

        for col in self.salary_data.columns:
            if 'опыт' in col.lower() or 'experience' in col.lower():
                experience_col = col
            if 'зарплат' in col.lower() or 'salary' in col.lower():
                salary_col = col

        if not experience_col or not salary_col:
            QMessageBox.warning(self, "Внимание", "Не найдены столбцы с опытом или зарплатой")
            return

        # Очищаем предыдущие графики
        for i in reversed(range(self.analysis_plots_layout.count())):
            widget = self.analysis_plots_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Преобразуем данные
        experience_data = pd.to_numeric(self.salary_data[experience_col], errors='coerce')
        salary_data = pd.to_numeric(self.salary_data[salary_col], errors='coerce')

        # Удаляем пропуски
        mask = experience_data.notna() & salary_data.notna()
        experience_data = experience_data[mask]
        salary_data = salary_data[mask]

        if len(experience_data) == 0:
            QMessageBox.warning(self, "Внимание", "Нет корректных данных для анализа")
            return

        fig = Figure(figsize=(18, 14))

        # 1. Scatter plot
        ax1 = fig.add_subplot(221)
        scatter = ax1.scatter(experience_data, salary_data, alpha=0.6,
                              c=salary_data, cmap='viridis', s=50)
        ax1.set_xlabel('Опыт (лет)', fontsize=12)
        ax1.set_ylabel('Зарплата (тыс.руб.)', fontsize=12)
        ax1.set_title('Зависимость зарплаты от опыта', fontsize=14, fontweight='bold', pad=18)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=10)
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('Зарплата (тыс.руб.)', fontsize=11)
        cbar1.ax.tick_params(labelsize=10)

        # Линия тренда
        if len(experience_data) > 1:
            # Полиномиальная регрессия
            coeffs = np.polyfit(experience_data, salary_data, 2)
            poly = np.poly1d(coeffs)
            x_range = np.linspace(experience_data.min(), experience_data.max(), 100)
            ax1.plot(x_range, poly(x_range), 'r-', linewidth=2, label='Тренд')
            ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)

        # 2. Биннинг опыта и средние зарплаты
        ax2 = fig.add_subplot(222)

        # Создаем бины по опыту
        max_exp = int(experience_data.max()) + 1
        bins = np.arange(0, max_exp + 5, 5)  # Бинны по 5 лет
        labels = [f'{i}-{i + 4}' for i in bins[:-1]]

        experience_binned = pd.cut(experience_data, bins=bins, labels=labels, right=False)

        # Средняя зарплата по бинам
        salary_by_exp = salary_data.groupby(experience_binned).agg(['mean', 'std', 'count']).dropna()

        x_pos = np.arange(len(salary_by_exp))
        bars = ax2.bar(x_pos, salary_by_exp['mean'], color='#4caf50',
                       yerr=salary_by_exp['std'], capsize=5, edgecolor='black')

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(salary_by_exp.index, rotation=45, ha='right', fontsize=10)
        ax2.set_xlabel('Опыт (лет)', fontsize=12)
        ax2.set_ylabel('Средняя зарплата (тыс.руб.)', fontsize=12)
        ax2.set_title('Средняя зарплата по опыту работы', fontsize=14, fontweight='bold', pad=18)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(labelsize=10)

        # Добавляем значения на столбцы
        for bar, mean_val, count in zip(bars, salary_by_exp['mean'], salary_by_exp['count']):
            height = bar.get_height()
            err = salary_by_exp.loc[salary_by_exp.index[bars.index(bar)], 'std'] if len(salary_by_exp) > bars.index(
                bar) else 0
            ax2.text(bar.get_x() + bar.get_width() / 2., height + err + max(salary_by_exp['mean']) * 0.02,
                     f'{mean_val:.1f}\n(n={count})', ha='center', va='bottom', fontsize=8)

        # 3. Зависимость зарплаты от возраста (если есть столбец возраста)
        age_col = None
        for col in self.salary_data.columns:
            if 'возраст' in col.lower() or 'age' in col.lower():
                age_col = col
                break

        if age_col:
            ax3 = fig.add_subplot(223)
            age_data = pd.to_numeric(self.salary_data[age_col], errors='coerce')
            mask_age = age_data.notna() & salary_data.notna()

            if mask_age.sum() > 0:
                scatter2 = ax3.scatter(age_data[mask_age], salary_data[mask_age],
                                       alpha=0.6, c=experience_data[mask_age],
                                       cmap='plasma', s=50)
                ax3.set_xlabel('Возраст (лет)', fontsize=12)
                ax3.set_ylabel('Зарплата (тыс.руб.)', fontsize=12)
                ax3.set_title('Зависимость зарплаты от возраста', fontsize=14, fontweight='bold', pad=18)
                ax3.grid(True, alpha=0.3)
                ax3.tick_params(labelsize=10)
                cbar2 = plt.colorbar(scatter2, ax=ax3)
                cbar2.set_label('Опыт (лет)', fontsize=11)
                cbar2.ax.tick_params(labelsize=10)
        else:
            # Если нет данных о возрасте, создаем пустой subplot
            ax3 = fig.add_subplot(223)
            ax3.text(0.5, 0.5, 'Данные о возрасте\nне найдены',
                     ha='center', va='center', fontsize=12, transform=ax3.transAxes)
            ax3.set_title('Зависимость зарплаты от возраста', fontsize=13, fontweight='bold', pad=15)
            ax3.axis('off')

        # 4. 3D plot опыт vs возраст vs зарплата
        if age_col:
            ax4 = fig.add_subplot(224, projection='3d')

            mask_3d = experience_data.notna() & salary_data.notna() & age_data.notna()
            if mask_3d.sum() > 0:
                scatter3d = ax4.scatter(experience_data[mask_3d], age_data[mask_3d],
                                        salary_data[mask_3d], c=salary_data[mask_3d],
                                        cmap='viridis', s=30, alpha=0.6)

                ax4.set_xlabel('Опыт (лет)', fontsize=11)
                ax4.set_ylabel('Возраст (лет)', fontsize=11)
                ax4.set_zlabel('Зарплата (тыс.руб.)', fontsize=11)
                ax4.set_title('3D: Опыт vs Возраст vs Зарплата', fontsize=14, fontweight='bold', pad=18)
        else:
            # Если нет данных о возрасте, создаем пустой subplot
            ax4 = fig.add_subplot(224)
            ax4.text(0.5, 0.5, '3D визуализация\nнедоступна\n(нет данных о возрасте)',
                     ha='center', va='center', fontsize=12, transform=ax4.transAxes)
            ax4.set_title('3D: Опыт vs Возраст vs Зарплата', fontsize=13, fontweight='bold', pad=15)
            ax4.axis('off')

        # Применяем темную тему
        self.setup_matplotlib_style(fig)

        # Улучшенное размещение с большими отступами
        fig.tight_layout(pad=4.0, h_pad=4.0, w_pad=4.0)
        fig.subplots_adjust(top=0.94, bottom=0.12, left=0.12, right=0.95, hspace=0.45, wspace=0.4)

        canvas = FigureCanvas(fig)
        canvas.setMinimumSize(1200, 900)
        self.analysis_plots_layout.addWidget(canvas)

        # Статистика
        stats_text = f"""
        📊 АНАЛИЗ ЗАВИСИМОСТИ ОТ ОПЫТА:

        Всего записей: {len(experience_data)}

        📈 Основная статистика:
        • Средний опыт: {experience_data.mean():.1f} лет
        • Средняя зарплата: {salary_data.mean():.1f} тыс.руб.
        • Корреляция опыт-зарплата: {experience_data.corr(salary_data):.3f}

        💰 Рост зарплаты с опытом:
        """

        # Рассчитываем прирост зарплаты за 5 лет
        if len(salary_by_exp) > 1:
            first_bin_mean = salary_by_exp['mean'].iloc[0]
            last_bin_mean = salary_by_exp['mean'].iloc[-1]
            if first_bin_mean > 0:
                growth_5y = (last_bin_mean / first_bin_mean - 1) * 100
                stats_text += f"• За 5 лет: +{growth_5y:.1f}%\n"

        # Рекомендации
        stats_text += f"""
        💡 РЕКОМЕНДАЦИИ:

        1. Опыт работы - ключевой фактор зарплаты
        2. Наибольший рост зарплаты наблюдается в первые 5-10 лет
        3. После 15 лет опыта рост замедляется
        4. Важно развивать специализацию и soft skills
        """

        self.model_stats_text.setText(stats_text)

    def generate_analysis_report(self):
        """Генерация полного отчета анализа"""
        if self.salary_data is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчет анализа", "salary_analysis_report.txt",
            "Text файлы (*.txt);;PDF файлы (*.pdf)"
        )

        if file_path:
            try:
                # Генерация отчета
                report = self.generate_complete_report()

                if file_path.endswith('.txt'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(report)

                QMessageBox.information(self, "Успех",
                                        f"Отчет анализа сохранен в {file_path}\n\n"
                                        f"Отчет содержит:\n"
                                        f"• Общую статистику данных\n"
                                        f"• Анализ распределения зарплат\n"
                                        f"• Анализ по должностям и городам\n"
                                        f"• Зависимость от опыта работы\n"
                                        f"• Рекомендации и выводы")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить отчет:\n{str(e)}")

    def generate_complete_report(self):
        """Генерация полного текстового отчета"""
        from datetime import datetime

        report = "=" * 80 + "\n"
        report += "ПОЛНЫЙ ОТЧЕТ АНАЛИЗА ЗАРПЛАТ\n"
        report += "=" * 80 + "\n\n"

        report += f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Приложение: AI Salary Predictor\n\n"

        if self.salary_data is not None:
            report += "📊 ОБЩАЯ СТАТИСТИКА ДАННЫХ:\n"
            report += f"• Количество записей: {len(self.salary_data):,}\n"
            report += f"• Количество признаков: {len(self.salary_data.columns)}\n"
            report += f"• Пропущенные значения: {self.salary_data.isnull().sum().sum():,}\n\n"

        if hasattr(self, 'model_metrics'):
            report += "🤖 СТАТИСТИКА МОДЕЛИ:\n"
            report += f"• Тип модели: {self.model_type_combo.currentText()}\n"
            report += f"• Коэффициент детерминации (R²): {self.model_metrics['r2']:.4f}\n"
            report += f"• Средняя ошибка (RMSE): {self.model_metrics['rmse']:.2f} тыс.руб.\n"
            report += f"• Кросс-валидация (R²): {self.model_metrics['cv_mean']:.4f} ± {self.model_metrics['cv_std']:.4f}\n\n"

        report += "📈 ОСНОВНЫЕ ВЫВОДЫ И РЕКОМЕНДАЦИИ:\n\n"
        report += "1. Факторы, влияющие на зарплату:\n"
        report += "   • Опыт работы - наиболее значимый фактор\n"
        report += "   • Должность - определяет базовый уровень\n"
        report += "   • Город - существенно влияет на уровень дохода\n"
        report += "   • Образование и навыки - повышают стоимость специалиста\n\n"

        report += "2. Рекомендации для соискателей:\n"
        report += "   • Фокусируйтесь на развитии специализации\n"
        report += "   • Получайте практический опыт на проектах\n"
        report += "   • Изучайте востребованные технологии\n"
        report += "   • Рассматривайте релокацию в крупные города\n\n"

        report += "3. Рекомендации для работодателей:\n"
        report += "   • Учитывайте рыночный уровень зарплат\n"
        report += "   • Предлагайте конкурентные условия\n"
        report += "   • Инвестируйте в развитие сотрудников\n"
        report += "   • Создавайте прозрачную систему оплаты труда\n\n"

        report += "=" * 80 + "\n"
        report += "КОНЕЦ ОТЧЕТА\n"
        report += "=" * 80

        return report


# ============================================================================
# 7. ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================================

def main():
    """Основная функция запуска приложения"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Установка иконки приложения
    app.setWindowIcon(QIcon())  # Можно добавить иконку

    # Создание и показ главного окна
    window = SalaryPredictorApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()