import sort
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input,
    Conv2DTranspose,
    concatenate,
    Activation,
    MaxPooling2D,
    Conv2D,
    BatchNormalization,
)
from tensorflow.keras.models import Model
import streamlit as st  # pip install streamlit
import pandas as pd  # pip install pandas
import numpy as np
import os
import shutil
import glob
import random
from time import time
import tempfile
import zipfile

from random import randint
import cv2  # OpenCV Library

import streamlit as st

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# from keras.utils.vis_utils import plot_model


# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# Параметры обучения нейросети
IMG_W = 640  # Ширина картинки
IMG_H = 360  # Высота картинки

# Цвета пикселов сегментированных изображений BGR
CARPET0 = (255, 0, 0)  # Ковер 60*90 (синий)
CARPET1 = (0, 255, 0)  # Ковер 85*150 (зеленый)
CARPET2 = (0, 255, 255)  # Ковер 115*200 (желтый)
CARPET3 = (0, 0, 255)  # Ковер 150*300 (красный)
CARPET4 = (255, 0, 255)  # Ковер 115*400 (фиолетовый)
CARPET5 = (255, 255, 0)  # Ковер х*х (желтый)

OTHER = (0, 0, 0)  # Остальное (черный)

CLASS_LABELS = (OTHER, CARPET0, CARPET1, CARPET2, CARPET3, CARPET4)
CLASS_COUNT = len(CLASS_LABELS)  # Количество классов на изображении

# Номер стартового кадра
Started_Frame = 3505
# Количество кадров для обработки
Frames_Limit = 1000

# Путь к файлу с весами обученной модели
path_to_weights = "/home/mikz/flying-carpets/unet_carpets_weights_Left_19_40_13-05"

# Путь к видеофайлу для обработки
path_to_video = "Video/Batch_CP_16_10.mp4"

# Разрешение видео для отображения
MAX_WIDTH = 640
MAX_HEIGHT = 360
file_uploaded = None

# Глобальные переменные Streamlit
if "visibility" not in st.session_state:
    st.session_state.model_stage = "visible"
    st.session_state.summary = "collapsed"
    st.session_state.progressbar = "hidden"
    st.session_state.model = None
    st.session_state.video = None
if "slider_side" not in st.session_state:
    st.session_state.slider_side = (0,Frames_Limit)
if 'prev_frame' not in st.session_state:
    st.session_state.prev_frame = None
if 'mot_tracker' not in st.session_state:
    st.session_state.mot_tracker = None
    st.session_state.track = None

# Модель нейронной сети

def unet(class_count, input_shape):  # количество классов  # форма входного изображения
    # Создаем входной слой формой input_shape
    img_input = Input(input_shape)

    #    ''' Block 1 '''
    x = Conv2D(64, (3, 3), padding="same", name="block1_conv1")(
        img_input
    )  # Добавляем Conv2D-слой с 64-нейронами
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    x = Conv2D(64, (3, 3), padding="same", name="block1_conv2")(
        x
    )  # Добавляем Conv2D-слой с 64-нейронами
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation и запоминаем в переменной block_1_out
    block_1_out = Activation("relu")(x)

    # Добавляем слой MaxPooling2D
    x = MaxPooling2D()(block_1_out)

    #    ''' Block 2 '''
    x = Conv2D(128, (3, 3), padding="same", name="block2_conv1")(
        x
    )  # Добавляем Conv2D-слой с 128-нейронами
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    x = Conv2D(128, (3, 3), padding="same", name="block2_conv2")(
        x
    )  # Добавляем Conv2D-слой с 128-нейронами
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation и запоминаем в переменной block_2_out
    block_2_out = Activation("relu")(x)

    # Добавляем слой MaxPooling2D
    x = MaxPooling2D()(block_2_out)

    #    ''' Block 3 '''
    x = Conv2D(256, (3, 3), padding="same", name="block3_conv1")(
        x
    )  # Добавляем Conv2D-слой с 256-нейронами
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    x = Conv2D(256, (3, 3), padding="same", name="block3_conv2")(
        x
    )  # Добавляем Conv2D-слой с 256-нейронами
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    x = Conv2D(256, (3, 3), padding="same", name="block3_conv3")(
        x
    )  # Добавляем Conv2D-слой с 256-нейронами
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation и запоминаем в переменной block_3_out
    block_3_out = Activation("relu")(x)

    # Добавляем слой MaxPooling2D
    x = MaxPooling2D()(block_3_out)

    #   ''' Block 4 '''
    x = Conv2D(512, (3, 3), padding="same", name="block4_conv1")(
        x
    )  # Добавляем Conv2D-слой с 512-нейронами
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    x = Conv2D(512, (3, 3), padding="same", name="block4_conv2")(
        x
    )  # Добавляем Conv2D-слой с 512-нейронами
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    x = Conv2D(512, (3, 3), padding="same", name="block4_conv3")(
        x
    )  # Добавляем Conv2D-слой с 512-нейронами
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation и запоминаем в переменной block_4_out
    block_4_out = Activation("relu")(x)
    x = block_4_out

    #   ''' UP 2 '''
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(
        x
    )  # Добавляем слой Conv2DTranspose с 256 нейронами
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    # Объединяем текущий слой со слоем block_3_out
    x = concatenate([x, block_3_out])
    # Добавляем слой Conv2D с 256 нейронами
    x = Conv2D(256, (3, 3), padding="same")(x)
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    x = Conv2D(256, (3, 3), padding="same")(x)
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    #   ''' UP 3 '''
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(
        x
    )  # Добавляем слой Conv2DTranspose с 128 нейронами
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    # Объединяем текущий слой со слоем block_2_out
    x = concatenate([x, block_2_out])
    # Добавляем слой Conv2D с 128 нейронами
    x = Conv2D(128, (3, 3), padding="same")(x)
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    # Добавляем слой Conv2D с 128 нейронами
    x = Conv2D(128, (3, 3), padding="same")(x)
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    #   ''' UP 4 '''
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(
        x
    )  # Добавляем слой Conv2DTranspose с 64 нейронами
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    # Объединяем текущий слой со слоем block_1_out
    x = concatenate([x, block_1_out])
    # Добавляем слой Conv2D с 64 нейронами
    x = Conv2D(64, (3, 3), padding="same")(x)
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    # Добавляем слой Conv2D с 64 нейронами
    x = Conv2D(64, (3, 3), padding="same")(x)
    # Добавляем слой BatchNormalization
    x = BatchNormalization()(x)
    # Добавляем слой Activation
    x = Activation("relu")(x)

    # Добавляем Conv2D-Слой с softmax-активацией на class_count-нейронов
    x = Conv2D(class_count, (3, 3), activation="softmax", padding="same")(x)

    # Создаем модель с входом 'img_input' и выходом 'x'
    model = Model(img_input, x)

    # Компилируем модель
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    # Возвращаем сформированную модель
    return model

# Функция преобразования тензора меток класса в цветное сегметрированное изображение
def labels_to_rgb(image_list  # список одноканальных изображений
                ):

    result = []

    # Для всех картинок в списке:
    for y in image_list:
        # Создание пустой цветной картики
        temp = np.zeros((IMG_H, IMG_W, 3), dtype="uint8")

        # По всем классам:
        for i, cl in enumerate(CLASS_LABELS):
            # Нахождение пикселов класса и заполнение цветом из CLASS_LABELS[i]
            temp[np.where(np.all(y == i, axis=-1))] = CLASS_LABELS[i]

        result.append(temp)

    return np.array(result)


# Функция преобразования одного цвета цветной маски в двоичную


def monochomize_prediction(predicion, class_number):  # список цветных изображений
    sample = np.array(predicion)

    # Создание пустой 1-канальной
    mono_prediction = np.zeros((sample.shape[0], sample.shape[1], 1), dtype="uint8")
    mono_prediction[np.where(np.all(sample == class_number, axis=-1))] = 1

    return mono_prediction

# Функция детектора на основе openCV
def ellipses_detector_openCV(
    mono_prediction,
    # Минимальная площадь вписанного эллипса (возможно достаточно будет второго параметра)
    min_area=1000,
    min_cnt_len=10,  # Если контур содержит мало точек, то в него нельзя вписать эллипс
):
    contours, _ = cv2.findContours(
        mono_prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    Coords = []
    Areas = []
    for cnt in contours:
        if len(cnt) > min_cnt_len:
            ellipse = cv2.fitEllipse(cnt)
            ellipse_area = ellipse[1][0] * ellipse[1][1]
            if ellipse_area > min_area:
                Coords.append([ellipse[0][0], ellipse[0][1]])
                Areas.append(ellipse_area)
    if len(Areas) != 0:
        # Сортирую контуры по площади вписанного эллипса и вычисляю соответствующий индекс
        index = Areas.index(sorted(Areas, reverse=True)[0])
        return Coords[index], Areas[index]
    else:
        return [0, 0], 0


# Функция анализа предсказания
def Analyze_Prediction(prediction):
    Coords = []
    Areas = []
    DetectedClass = []
    for cls in range(1, CLASS_COUNT):
        Coord, Area = ellipses_detector_openCV(monochomize_prediction(prediction, cls))
        if Area != 0:
            Coords.append(Coord)
            Areas.append(Area)
            DetectedClass.append(cls)
        if len(Areas) != 0:
            index = Areas.index(sorted(Areas, reverse=True)[0])
            return Coords[index], DetectedClass[index]
        else:
            return [0, 0], 0
        
# Функция отображения координат на изображении
def draw_colored_coords(image, coords, color, img_w, img_h, thickness=1):
    cv2.line(image, (0, coords[1]), (img_w, coords[1]), color, thickness)
    cv2.line(image, (coords[0], 0), (coords[0], img_h), color, thickness)
    return image

# Детектор для видео
def detect_carpets(frame):
    min_area = 500
    bbox = []
    if st.session_state.prev_frame != None:
        diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY),)
        diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
        img_for_predict = cv2.resize(diff, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        predict = np.argmax(model.predict(np.array([img_for_predict]), verbose=0), axis=-1)
        mono_predict = monochomize_prediction(predict, CLASS_COUNT)
        contours, _ = cv2.findContours(mono_prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if w*h > min_area:
                bbox.append([x, y, x+w, y+h])        #Координаты и предсказания для трекера
        cls ,count = np.unique(predict, return_counts=True)
    st.session_state.prev_frame = frame
    return cls,counts,bbox

def track_carpets(bbox):
    if not st.session_state.mot_tracker:
        st.session_state.mot_tracker = Sort() #create instance of the SORT tracker
    for box in bbox:
        x1 = box[0]-box[2]/2
        x2 = box[0]+box[2]/2
        y1 = box[1]-box[3]/2
        y2 = box[1]+box[3]/2
        st.session_state.track = mot_tracker.update(np.array([x1, y1, x2, y2,-1]))
        return st.session_state.track[-1][0]           #return the ID of the tracked object
@st.cache_resource
def load_model():
    # Создание модели и вывод сводки по архитектуре
    if st.session_state.weight_uploaded:
        path_to_weights = st.session_state.weight_uploaded.name[:-3] + "h5"
        with zipfile.ZipFile(st.session_state.weight_uploaded, "r") as z:
            z.extractall(".")
            # Загрузка весов модели
            st.session_state.model.load_weights(path_to_weights)
            # st.session_state.model.summary(print_fn=lambda x: st.sidebar.text(x))
            st.sidebar.write(f"Модель {path_to_weights} успешно загруженна")
            return st.session_state.model

    else:
        print(f"Модель не загружена {st.session_state.weight_uploaded.name}")
        return


def set_video_range(filename):
    cap = cv2.VideoCapture(filename)
    if cap.isOpened() == False:
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.slider_side[0])
        print(st.session_state.slider_side[0])
        ret, frame = cap.read()
        if ret:
            image.image(frame)
        else:
            print("не найден кадр {frame}")
        cap.release()
    else:
        col1.write("не могу открыть файл {filename}")

def load_video():
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.slider_side[0])
    Frame_limit = st.session_state.slider_side[1]- st.session_state.slider_side[0]
    st.session_state.proggressbar = "visible"
    pb.progress(0, text="Video processing")
    # Вычисление переменных для масштабирования видео
    Vscale = MAX_WIDTH / width
    Hscale = MAX_HEIGHT / height
    if Vscale < Hscale:
        Scale_Coef = Vscale
    else:
        Scale_Coef = Hscale
    player_width = int(width * Scale_Coef)
    player_height = int(height * Scale_Coef)

    # Чтение кадра в качестве предыдущего для вычисления разницы
    ret, prevframe = cap.read()

    # Счетчик обработанных кадров
    frame_count = 0

    while True:
        # Чтение следующего кадра
        ret, frame = cap.read()

        # Подсчет обработанных кадров для прерывания цикла
        frame_count += 1
        if frame_count > Frames_Limit:
            break
        pb.progress(
            frame_count / Frames_Limit,
            text=f"Video processing{frame_count}/{Frames_Limit}",
        )
        if ret:
            cls,cnt,bbox = detect_carpets(frame)
        if False: '''
        if ret:
            t_0 = time()
            diff = cv2.absdiff(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY),
            )
            # image.image(diff)
            # Предсказание сегментационной сетью
            img_for_predict = cv2.resize(
                diff, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR
            )
            predict = np.argmax(
                model.predict(np.array([img_for_predict]), verbose=0), axis=-1
            )
            cls ,count = np.unique(predict, return_counts=True)
            stext60x90.write(f'Class:{cls} Count:{count}')
            
            # stext.write(predict)
            colored_predict = labels_to_rgb(predict[..., None])[0]

            # Извлечение класса области максимального размера и координат вписываемого в него эллипса
            DetectedCoords, DetectedClass = Analyze_Prediction(predict[..., None][0])
            # Преобразую картинки для вывода на экран

            prevframe = cv2.resize(
                prevframe, (player_width, player_height), interpolation=cv2.INTER_LINEAR
            )
            diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

            diff = cv2.resize(
                diff, (player_width, player_height), interpolation=cv2.INTER_LINEAR
            )
            colored_predict = labels_to_rgb(predict[..., None])[0]
            colored_predict = cv2.resize(
                colored_predict,
                (player_width, player_height),
                interpolation=cv2.INTER_LINEAR,
            )

            # Если были обнаружены координаты, рисую перекрестие
            if DetectedClass != 0:
                resized_detected_coords = (
                    DetectedCoords
                    * np.array([player_width / IMG_W, player_height / IMG_H])
                ).astype(int)

                # Рисую разметку координат на кадрах
                prevframe = draw_colored_coords(
                    prevframe,
                    resized_detected_coords,
                    CLASS_LABELS[DetectedClass],
                    player_width,
                    player_height,
                )
                diff = draw_colored_coords(
                    diff,
                    resized_detected_coords,
                    CLASS_LABELS[DetectedClass],
                    player_width,
                    player_height,
                )
                colored_predict = draw_colored_coords(
                    colored_predict,
                    resized_detected_coords,
                    CLASS_LABELS[DetectedClass],
                    player_width,
                    player_height,
                )

            # Объединяю кадры
            line_1st = np.hstack((prevframe, diff))
            line_2st = np.hstack((colored_predict, prevframe))
            result = np.vstack((line_1st, line_2st))

            t_f = time() - t_0
            time_text = str(round(1000 * t_f % 60)) + " мс."

            # clear_output(wait=True)
            print("Кадр", Started_Frame + frame_count, "обработан за", time_text)
            
            if DetectedClass != 0:
                print(
                    "Относительные координаты ковра:",
                    [DetectedCoords[0] / IMG_W, DetectedCoords[1] / IMG_H],
                )
                print("Класс:", DetectedClass)
            else:
                print("Ковер не обнаружен.\n")
            '''
            result = frame
            image.image(result)

            prevframe = frame

        else:
            break
    cap.release()


# create interface
st.set_page_config(page_title="Подсчет ковров", layout="wide")
st.title("Подсчет ковров")

st.session_state.weight_uploaded = st.sidebar.file_uploader(
    "Веса модели", type="zip", label_visibility=st.session_state.model_stage
)

main = st.empty()
pb = st.sidebar.empty()
if st.session_state.weight_uploaded == None:
    st.session_state.model_stage = "visible"
    main.write("Загрузите веса модели!")
    st.stop()
else:
    st.session_state.video = st.sidebar.file_uploader("загрузите видео", type="mp4")
    main.empty()
    st.session_state.model_stage = "hidden"
    if st.session_state.model == None:
        st.session_state.model = unet(CLASS_COUNT, (IMG_H, IMG_W, 1))
    model = load_model()
    col1, col2 = main.columns([3, 1], gap="small")
    col1.subheader("Видео")
    col2.subheader("Счетчик ковров")
    image = col1.empty()
    stext = col2.container()
    stext60x90 = stext.empty()
    stext85x150 = stext.empty()
    stext115x200 = stext.empty()
    stext150x300 = stext.empty()
    stext115x400 = stext.empty()
    stext_All = stext.empty()
    stext60x90.write("  60x90:")
    stext85x150.write(" 85x150:")
    stext115x200.write("115x200:")
    stext150x300.write("150x300:")
    stext115x400.write("115x400:")
    stext_All.write("  Всего: 0")
if st.session_state.video:
    video = st.session_state.video
    st.session_state.summary = "collapsed"
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())
    tfile.close()
    cap = cv2.VideoCapture(tfile.name)
    print(cap)
    if cap.isOpened() == False:
        print(f"File:{tfile.name} not found")
        st.stop()
    frame_All = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.slider_side[0])
    col1.write(f"{video.name}: {st.session_state.slider_side[0]}")
    ret, frame = cap.read()
    if ret:
        image.image(frame)
    col1.slider(
        " ",
        0,
        frame_All,
        (0, frame_All),
        key="slider_side",
        on_change=set_video_range,
        args=[tfile.name],
    )
    frame_all = st.session_state.slider_side[1] - st.session_state.slider_side[0]
    st.sidebar.write(f"{video.name}: {frame_all} кадров")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if col1.button("Обработать"):
        load_video()
    st.stop()
