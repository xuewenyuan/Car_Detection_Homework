import cv2
import numpy as np
from car_detector.detector import car_detector, bow_features
from car_detector.pyramid import pyramid
from car_detector.non_maximum import non_max_suppression_fast as nms
from car_detector.sliding_window import sliding_window
import urllib

def in_range(number, test, thresh=0.2):
  return abs(number - test) < thresh

#test_image = "test.bmp"
img_path = "./test"
#remote = "http://previews.123rf.com/images/aremac/aremac0903/aremac090300044/4545419-Lonely-car-on-an-empty-parking-lot-Stock-Photo.jpg"
#urllib.urlretrieve(test_image, img_path)

svm, extractor = car_detector()
detect = cv2.xfeatures2d.SIFT_create()

w, h = 100, 40
#img = cv2.imread(img_path)
for i in range(170):
    print "test: ", i
    img_path = "%s/%s%d.bmp"  % ( "./test","test-",i)
    img = cv2.imread(img_path)

    rectangles = []
    counter = 1
    scaleFactor = 1.25
    scale = 1
    font = cv2.FONT_HERSHEY_PLAIN

    for resized in pyramid(img, scaleFactor):
      scale = float(img.shape[1]) / float(resized.shape[1])
      for (x, y, roi) in sliding_window(resized, 20, (100, 40)):

        if roi.shape[1] != w or roi.shape[0] != h:
          continue

        try:
          bf = bow_features(roi, extractor, detect)
          _, result = svm.predict(bf)
          a, res = svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT | cv2.ml.STAT_MODEL_UPDATE_MODEL)
          #print "Class: %d, Score: %f, a: %s" % (result[0][0], res[0][0], res)
          score = res[0][0]
          if result[0][0] == 1:
            if score < -1.0:
              rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x+w) * scale), int((y+h) * scale)
              rectangles.append([rx, ry, rx2, ry2, abs(score)])
        except:
          pass

        counter += 1

    windows = np.array(rectangles)
    boxes = nms(windows, 0.25)

    num = 0
    """
    _score = 0.0
    _x = 0.0
    _y = 0.0
    _x2 = 0.0
    _y2 = 0.0
    """
    for (x, y, x2, y2, score) in boxes:
      if num > 2:
          break
      print "box: ", num
      print x, y, x2, y2, score
      num += 1
      cv2.rectangle(img, (int(x),int(y)),(int(x2), int(y2)),(0, 255, 0), 1)
      cv2.putText(img, "%f" % score, (int(x),int(y)), font, 1, (0, 255, 0))
      """
      if score > _score:
          _x = x
          _y = y
          _x2 = x2
          _y2 = y2
          _score = score
    #print "test: ", i
    #print _x, _y, _x2, _y2, _score

    cv2.rectangle(img, (int(_x),int(_y)),(int(_x2), int(_y2)),(0, 255, 0), 1)
    cv2.putText(img, "%f" % _score, (int(_x),int(_y)), font, 1, (0, 255, 0))
    """


    cv2.imshow("img", img)
    cv2.waitKey(0)
    """
    if i == 8:
        cv2.imwrite("./test_result_8.bmp", img)
    if i == 17:
        cv2.imwrite("./test_result_17.bmp", img)
    if i == 31:
        cv2.imwrite("./test_result_31.bmp", img)
    """
    pass

"""
img = cv2.imread(test_image)

rectangles = []
counter = 1
scaleFactor = 1.25
scale = 1
font = cv2.FONT_HERSHEY_PLAIN

for resized in pyramid(img, scaleFactor):
  scale = float(img.shape[1]) / float(resized.shape[1])
  for (x, y, roi) in sliding_window(resized, 20, (100, 40)):

    if roi.shape[1] != w or roi.shape[0] != h:
      continue

    try:
      bf = bow_features(roi, extractor, detect)
      _, result = svm.predict(bf)
      a, res = svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT | cv2.ml.STAT_MODEL_UPDATE_MODEL)
      print "Class: %d, Score: %f, a: %s" % (result[0][0], res[0][0], res)
      score = res[0][0]
      if result[0][0] == 1:
        if score < -1.0:
          rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x+w) * scale), int((y+h) * scale)
          rectangles.append([rx, ry, rx2, ry2, abs(score)])
    except:
      pass

    counter += 1

windows = np.array(rectangles)
boxes = nms(windows, 0.25)

num = 0
_score = 0.0
_x = 0.0
_y = 0.0
_x2 = 0.0
_y2 = 0.0
for (x, y, x2, y2, score) in boxes:
  print "box: ", num
  print x, y, x2, y2, score
  num += 1
  if score > _score:
      _x = x
      _y = y
      _x2 = x2
      _y2 = y2

cv2.rectangle(img, (int(x),int(y)),(int(x2), int(y2)),(0, 255, 0), 1)
cv2.putText(img, "%f" % score, (int(x),int(y)), font, 1, (0, 255, 0))


cv2.imshow("img", img)
cv2.waitKey(0)
"""
