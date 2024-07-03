import cv2
path = "C:/University/Year2/Summer/Research/supg_video/supg/experiments/b_video.mp4"
# cap = cv2.VideoCapture("newout.mp4")
# 显示图片
# cap.set(cv2.CAP_PROP_POS_FRAMES, 8022)
# ret, frame = cap.read()
# cv2.imshow('Image', frame)
# cv2.waitKey(0)  # Wait for any key to be pressed before closing the window
# cv2.destroyAllWindows()

def get_frame(video_path: str, frame_number: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():  
            print("Error: Could not open video.")  

    success, frame = cap.read()  
    frame_count = 0  
    while success and frame_count < frame_number:  
        success, frame = cap.read()  
        frame_count += 1
        
    if success:  
            cv2.imshow('Frame', frame)  
            cv2.waitKey(0)  
    else:  
        print(f"Error: Frame {frame_number} could not be read.")

    cap.release()
    
def get_clip(video_path : str, target_frame: int, period: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():  
        print("Error: Could not open video.")  
        return
    
    frame_count = 0
    if period / 2 > target_frame:
          success, frame = cap.read()
    else:
        while True:  
            success, frame = cap.read()  
            if not success:  
                print("Error: Reached end of video before start frame.")  
                break  
            if frame_count >= target_frame - period / 2:  
                break  
            frame_count += 1

    time = 0

    while time <= period and success:  
        cv2.imshow('Frame', frame)  
          
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  
           
        success, frame = cap.read()  
        frame_count += 1
        time += 1

    cap.release()
    cv2.destroyAllWindows()

#get_frame(path, 1826)
get_clip(path, 1874, 100)


