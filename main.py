# 基本ライブラリ
import streamlit as st
import av
from PIL import Image, ImageDraw, ImageFont, ImageOps
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import numpy as np
from turn import get_ice_servers

import detection_model

# 利用するモデルをラジオボタンで選択
radio_model = st.sidebar.radio('利用するモデルを選んでください', detection_model.set_st_radio())

# 利用するモデルをセット
model= detection_model.select_model(radio_model)

# モデルのlayer、parameterをテーブルでサイドバー表示
st.sidebar.table(data=detection_model.df_set())

# 表示する映像を左右反転させるか
video_ckbox = st.sidebar.checkbox('映像の左右反転')

st.title('柿ピー検出')

# 推論と描写
def video_frame_callback(frame):
    #av.video.frame.VideoFrameからndarray型に変換
    img = frame.to_ndarray(format="bgr24")
    #ndarray型からPILに変換
    img = Image.fromarray(img)
    # video_ckboxがTrueなら映像を左右反転する
    img = ImageOps.mirror(img)  if video_ckbox else img

    #ネットワークの準備
    #img : 画像データ
    #conf : 確率のMIN値
    results = model(img, conf=0.7)

    #物体名を描画する
    font = ImageFont.truetype(font="ipaexg00401/ipaexg.ttf", size=10)  # フォントとサイズを指定する
    draw = ImageDraw.Draw(img)

    #CLASS_NAMES : 物体検出クラス
    #CLASS_COLORS : バウンディングボックスのカラー指定
    CLASS_NAMES = ['柿の種', 'ピーナッツ']
    CLASS_COLORS = [(15, 15, 255), (0, 128, 0)]

    # バウンディングボックスの描写
    for pred in results:

        # pred : tensor型
        # box : 位置
        # cls : 物体検出クラス
        # conf : 信頼率
        for box, cls, conf in zip(pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf):

            #class_int 0:柿の種 1:ピーナッツ
            class_int = int(cls.numpy())

            
            # バウンディングボックスを描写
            # xmin : 左上
            # ymin : 左下
            # xmax : 右上
            # ymax : 右下
            
            xmin, ymin, xmax, ymax = box.tolist()
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=CLASS_COLORS[class_int], width=5)
            

            # 物体名と確率を描く
            # label : 物体名
            # label_with_prob : 物体名と確率

            label = CLASS_NAMES[int(cls.numpy())]
            label_with_prob = f'{label} {conf:.2f}' #物体名と信頼率（小数点2まで）を表示
            non, non, w, h = font.getbbox(label_with_prob) # nonは利用しない。 getsizeでもできるが、Pillows==10.0.0は利用できなくなったので、getbboxで代用
            draw.rectangle([xmin, ymin, xmin+w, ymin+h], fill=CLASS_COLORS[class_int]) # 物体名を記載する枠を描画する
            draw.text((xmin, ymin), label_with_prob, fill="white", font=font)  # 物体名を描画する
    return av.VideoFrame.from_ndarray(np.array(img), format="bgr24")

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