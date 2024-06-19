import cv2
cap = cv2.VideoCapture("../../out.mp4")
# cap = cv2.VideoCapture("newout.mp4")
# 显示图片
# cap.set(cv2.CAP_PROP_POS_FRAMES, 8022)
# ret, frame = cap.read()
# cv2.imshow('Image', frame)
# cv2.waitKey(0)  # Wait for any key to be pressed before closing the window
# cv2.destroyAllWindows()

# 每5帧保留1帧，保存为一个新的视频文件
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('newout.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        if frame_count % 5 == 0:
            out.write(frame)
        frame_count += 1
    else:
        break

cap.release()
out.release()


