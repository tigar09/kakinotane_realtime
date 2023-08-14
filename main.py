# 基本ライブラリ
import streamlit as st
import av
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import cv2
from turn import get_ice_servers

import detection_model

#利用するモデルをラジオボタンで選択
radio_model = st.sidebar.radio('利用するモデルを選んでください', detection_model.set_st_radio())

#利用するモデルをセット
model= detection_model.select_model(radio_model)

st.sidebar.table(data=detection_model.df_set())

st.title('柿ピー検出')

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = Image.fromarray(img)

     #ネットワークの準備
    #img : 画像データ
    #conf : 確率のMIN値
    results = model(img, conf=0.7)

    #物体名を描画する
    font = ImageFont.truetype(font="ipaexg00401/ipaexg.ttf", size=60)  # フォントとサイズを指定する
    draw = ImageDraw.Draw(img)

    #class_names : 物体検出クラス
    CLASS_NAMES = ['柿の種', 'ピーナッツ']
    CLASS_COLORS = [(15, 15, 255), (0, 128, 0)]

    for pred in results:

        # pred : tensor型
        # box : 位置
        # cls : 物体検出クラス
        # conf : 確率
        for box, cls, conf in zip(pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf):

            #class_int 0:柿の種 1:ピーナッツ
            class_int = int(cls.numpy())

            
            # バウンディングボックスを描く
            # xmin : 左上
            # ymin : 左下
            # xmax : 右上
            # ymax : 右下
            
            xmin, ymin, xmax, ymax = box.tolist()
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=CLASS_COLORS[class_int], width=5)
            
            # '''
            # 物体名と確率を描く
            # label : 物体名
            # label_with_prob : 物体名と確率
            # '''
            label = CLASS_NAMES[int(cls.numpy())]
            label_with_prob = f'{label} {conf:.2f}'
            w, h = font.getsize(label_with_prob)
            draw.rectangle([xmin, ymin, xmin+w, ymin+h], fill=CLASS_COLORS[class_int])
            draw.text((xmin, ymin), label_with_prob, fill="white", font=font)  # 物体名を描画する
    return av.VideoFrame.from_ndarray(frame, format="bgr24")

# def video_frame_callback(frame):
#     img = frame.to_ndarray(format="bgr24")

#     # ここに、カメラ画像 img に対する処理を記述する
#     # 【サンプル】バウンディングボックスとラベルを表示する
#     BOXES = [(100, 100, 250, 250), (20, 20, 120, 120)]
#     LABELS = ["Object A", "Object B"]
#     COLORS = [(15, 15, 255), (0, 128, 0)]

#     for i in range(2):
#         xmin, ymin, xmax, ymax = BOXES[i]
#         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), COLORS[i], 2)
#         x = xmin + 5 if xmin + 5 < xmax - 5 else xmin
#         y = ymin - 10 if ymin - 10 > 15 else ymin + 15
#         cv2.putText(img, LABELS[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[i], 2)
#     # 処理ここまで

#     return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("Real-time video streaming")
st.caption("リアルタイムのカメラ画像を表示します")

webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)