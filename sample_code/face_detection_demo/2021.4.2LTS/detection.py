import cv2 as cv
import time

from openvino.inference_engine import IECore

face_xml = "./intel/face-detection-0205/FP16-INT8/face-detection-0205.xml"
face_bin = "./intel/face-detection-0205/FP16-INT8/face-detection-0205.bin"

ie = IECore()
for device in ie.available_devices:
    print(device)

# Read IR
net = ie.read_network(model=face_xml, weights=face_bin)

input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))

# 输入设置
n, c, h, w = net.input_info[input_blob].input_data.shape

# 设备关联推理创建
exec_net = ie.load_network(network=net, device_name="CPU")

# cap = cv.VideoCapture("./people-detection.mp4")
cap = cv.VideoCapture(0)
while True:

    inf_start = time.time()
    ret, src = cap.read()
    if ret is not True:
        break
    # 处理输入图象
    image = cv.resize(src, (w, h))
    image = image.transpose(2, 0, 1)

    # 推理
    prob = exec_net.infer(inputs={input_blob: [image]})

    # 后处理
    ih, iw, ic = src.shape
    res = prob["boxes"]
    for obj in res:
        if obj[4] > 0.5:
            xmin = int(obj[0] * iw / w)
            ymin = int(obj[1] * ih / h)
            xmax = int(obj[2] * iw / w)
            ymax = int(obj[3] * ih / h)
            cv.rectangle(src, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)
            cv.putText(src, str("%.3f" % obj[4]), (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 8)
    inf_end = time.time() - inf_start
    cv.putText(src, "infer time(ms): %.3f, FPS: %.2f" % (inf_end * 1000, 1 / (inf_end + 0.0001)), (10, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, 8)
    cv.imshow("face_detect", src)
    c = cv.waitKey(1)
    if c == 27: # ESC
        break
cv.destroyAllWindows()