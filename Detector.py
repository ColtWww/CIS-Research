import cv2

 
class BackgroundSubtractor():
    def __init__(self, detection_area, history):
        super(BackgroundSubtractor, self).__init__()
        self.detection_area = detection_area  #  检测区域
        self.history = history  # 设置训练帧数
        self.bs = self.build_subtractor(history=history)  # 建立背景差分器
        self.count= 0
 
    def build_subtractor(self, history):
        bs = cv2.createBackgroundSubtractorKNN(detectShadows=False)  # 背景减除器，设置阴影检测
        bs.setHistory(history)
        return bs
 
    def train(self, frame):  #  背景建模
        if self.count == self.history:
            return True
        else:
            fg_mask = self.bs.apply(frame)   # 获取 foreground mask
            self.count += 1
            return False
        
        
 
    def detect(self, frame):  #   检测
        fg_mask = self.bs.apply(frame)   # 获取 foreground mask
        car_count=0
        # 对原始帧进行膨胀去噪
        th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)), iterations=3)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
        # 获取所有检测框
        contours, hier = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for c in contours:
            # 获取矩形框边界坐标
            x, y, w, h = cv2.boundingRect(c)
            if y < self.detection_area[1] or x < self.detection_area[0] or y > self.detection_area[3]:continue  # 排除计数区域之外的目标干扰
            # 计算矩形框的面积
            area = cv2.contourArea(c)
            if 500 < area:
                boxes.append([x, y, w, h])
                car_count=car_count+1
                print(car_count)
        return boxes
       
 
#  测试
if __name__ == '__main__':
    camera = cv2.VideoCapture(1)
    video = camera
    res, frame = camera.read()
    detector = BackgroundSubtractor(detection_area = [100, int(frame.shape[0]/3),frame.shape[1], frame.shape[0]-50],history=500)
 
    while True:
            res, frame = camera.read()
            if not res:
                camera = cv2.VideoCapture(video)
                res, frame = camera.read()

                #break
            if not detector.train(frame):continue
            boxes = detector.detect(frame)
 
            #cv2.rectangle(frame, (100, int(frame.shape[0]/3)), (frame.shape[1], frame.shape[0]-70), (0, 255, 0), 2)
            for box in boxes:
                x,y,w,h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, 'Car', (x,y), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 2)
 
            cv2.imshow("detection", frame)
            if cv2.waitKey(30) & 0xff == 27:
                break
 
    camera.release()
    cv2.destroyAllWindows()
 