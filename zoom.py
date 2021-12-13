import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Initialize the Colors
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
lite = 0.6
dark = 1

# Initialize the Web cam
videoCapture = cv2.VideoCapture(0)
videoCapture.set(3, 1280)
videoCapture.set(4, 720)

# Initialize Hand tracking with Media Pipes
handDetector = HandDetector(detectionCon=0.7, maxHands=2)


class Hand:
    def __init__(self):
        self.landmarks = []
        self.center_point = []
        self.bbox = []


class Rectangle:
    def __init__(self, color=red, position=[1000, 100], size=[100, 100]):
        self.x, self.y = position
        self.original_w, self.original_h = size
        self.current_w, self.current_h = size
        self.original_color = color
        self.current_color = color
        self.alpha = lite
        self.sticky_counter = 0
        self.grab_state = False
        self.grab_hand = None
        self.grab_bbox_w = 1
        self.grab_bbox_h = 1
        self.grab_distance = None

    def update(self, right_hand, left_hand):
        landmarks = []
        if self.grab_state is True and self.grab_hand == "Right":
            landmarks = right_hand.landmarks
            bbox = right_hand.bbox
        elif self.grab_state is True and self.grab_hand == "Left":
            landmarks = left_hand.landmarks
            bbox = left_hand.bbox
        else:
            return False

        if not landmarks:
            return False

        self.x = landmarks[9][0] - 50
        self.y = landmarks[9][1] - 50

        print(bbox[2], bbox[3])
        print(self.grab_bbox_w, self.grab_bbox_h)
        print(self.current_w, self.current_h)
        self.current_w = int((self.original_w * bbox[2]) / self.grab_bbox_w)
        self.current_h = int((self.original_h * bbox[3]) / self.grab_bbox_h)
        print(bbox[2], bbox[3])
        print(self.grab_bbox_w, self.grab_bbox_h)
        print(self.current_w, self.current_h)
        return True

    def handover(self, right_hand, left_hand):
        if self.grab_state is True:
            return True
        if right_hand.landmarks:
            if self.x < right_hand.landmarks[9][0] < (self.x + self.current_w) and \
                    self.y < right_hand.landmarks[9][1] < (self.y + self.current_h):
                self.grab_hand = "Right"
                self.current_color = green
                self.alpha = dark
                if not self.grab_distance:
                    self.grab_distance = handDetector.findDistance(right_hand.landmarks[4],
                                                               right_hand.landmarks[20])[0]
                    print("grab disatane:", self.grab_distance)
                return True
        if left_hand.landmarks:
            if self.x < left_hand.landmarks[9][0] < (self.x + self.current_w) and \
                    self.y < left_hand.landmarks[9][1] < (self.y + self.current_h):
                self.grab_hand = "Left"
                self.current_color = green
                self.alpha = dark
                if not self.grab_distance:
                    self.grab_distance = handDetector.findDistance(left_hand.landmarks[4],
                                                               left_hand.landmarks[20])[0]
                    print("grab disatane:", self.grab_distance)
                return True

        if not self.grab_hand:
            self.grab_hand = None
            self.current_color = self.original_color
            self.alpha = lite
            self.grab_distance = None
            return False

    def grab(self, right_hand, left_hand):
        landmarks = []
        if self.grab_hand == "Right":
            landmarks = right_hand.landmarks
            bbox = right_hand.bbox
        elif self.grab_hand == "Left":
            landmarks = left_hand.landmarks
            bbox = left_hand.bbox
        else:
            return False

        if not landmarks:
            return False

        if handDetector.findDistance(landmarks[4], landmarks[20])[0] < int(self.grab_distance/2):
            self.alpha = dark
            self.sticky_counter = 3
            if not self.grab_state:
                self.grab_bbox_w = bbox[2]
                self.grab_bbox_h = bbox[3]
                print("grab bbox wh updated, ", self.grab_bbox_w, self.grab_bbox_h)
            self.grab_state = True
            return True
        elif self.sticky_counter > 0:
            self.grab_state = True
            self.sticky_counter = self.sticky_counter - 1
            return True
        else:
            self.grab_state = False
            self.grab_hand = None
            self.alpha = lite
            self.sticky_counter = 0
            #self.current_w = self.original_w
            #self.current_h = self.original_h
            self.grab_bbox_w = 1
            self.grab_bbox_h = 1
            return False


def display(img, rectangles):
    # overlay the rectangle
    image_with_shapes = img.copy()
    shapes = np.zeros_like(img, np.uint8)
    for rectangle in rectangles:
        cv2.rectangle(shapes, (rectangle.x, rectangle.y),
                      (rectangle.x + rectangle.current_w, rectangle.y + rectangle.current_h),
                      rectangle.current_color, cv2.FILLED)
        mask = shapes.astype(bool)
        image_with_shapes[mask] = cv2.addWeighted(img, 1,
                                                  shapes, rectangle.alpha, 0)[mask]

    # Display the image
    cv2.imshow('Image', image_with_shapes)
    cv2.waitKey(1)


def main():
    #rectangles = [Rectangle(size=[200, 200]), Rectangle(color=blue, position=[100, 100], size=[200, 200])]
    rectangles = [Rectangle(size=[200, 200])]
    while True:
        success, img = videoCapture.read()
        img = cv2.flip(img, 1)

        # Detect the hands
        hands, img = handDetector.findHands(img, draw=False, noBox=False), img

        left_hand = Hand()
        right_hand = Hand()

        if hands:
            for hand in hands:
                if hand["type"] == "Right":
                    left_hand.landmarks = hand["lmList"]
                    left_hand.center_point = hand['center']
                    left_hand.bbox = hand["bbox"]
                if hand["type"] == "Left":
                    right_hand.landmarks = hand["lmList"]
                    right_hand.center_point = hand['center']
                    right_hand.bbox = hand["bbox"]

            for rectangle in rectangles:
                if rectangle.handover(right_hand, left_hand):
                    if rectangle.grab(right_hand, left_hand):
                        rectangle.update(right_hand, left_hand)

        display(img, rectangles)


if __name__ == "__main__":
    main()
