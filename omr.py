'''
Code to read OMR Sheets using OpenCV
'''
import cv2
import numpy as np
from imutils import contours
import imutils
import argparse
import pytesseract as tesseract
from PIL import Image

def trans_OMR(img1, img2):
    '''
    Function to get the perspective of the OMR
    Inputs:img1: source image
           img2: template for perspective
    '''
    MIN_MATCH_COUNT = 10
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

    else:
        print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None

    print ("Extracted transform")
    cv2.namedWindow('Trans', cv2.WINDOW_NORMAL)
    cv2.imshow('Trans', dst)
    cv2.waitKey(0)
    return dst

def roi_extractor(img):
    '''
    Extract the ROI for details of a candidate and the answers marked by him
    '''
    _,img_inv = cv2.threshold(img,135,255,cv2.THRESH_BINARY_INV)
    kernel=np.ones((5,5))
    img_inv = cv2.morphologyEx(img_inv, cv2.MORPH_CLOSE, kernel)
    _, cnts, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    temp=np.zeros_like(img)
    cv2.drawContours(temp,cnts,-1,255,5)
    for i, cnt in enumerate(cnts):
        area.append(cv2.contourArea(cnt))
    idx = np.argsort(area)
    detail = cnts[idx[-1]]
    ans1 = cnts[idx[-2]]
    ans2 = cnts[idx[-3]]
    dc = cv2.boundingRect(detail)
    ac1 = cv2.boundingRect(ans1)
    ac2 = cv2.boundingRect(ans2)
    detl_roi = img[dc[1]+7:dc[1]+dc[3]-7, dc[0]+7:dc[0]+dc[2]-7]
    if ac1[0] > ac2[0]:
        ans1_roi=img[ac2[1]+7:ac2[1]+ac2[3]-7,ac2[0]+7:ac2[0]+ac2[2]-7]
        ans2_roi=img[ac1[1]+7:ac1[1]+ac1[3]-7,ac1[0]+7:ac1[0]+ac1[2]-7]
    else:
        ans1_roi=img[ac1[1]+7:ac1[1]+ac1[3]-7,ac1[0]+7:ac1[0]+ac1[2]-7]
        ans2_roi=img[ac2[1]+7:ac2[1]+ac2[3]-7,ac2[0]+7:ac2[0]+ac2[2]-7]

    print ("Extracted Region of Interest")
    cv2.namedWindow('ROI',cv2.WINDOW_NORMAL)
    cv2.imshow('ROI',temp)
    cv2.waitKey(0)
    return detl_roi,ans1_roi,ans2_roi

def bubble_check(warped,bubbled_response):
    thresh = cv2.threshold(warped, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    print ("Displaying Threshold Map")
    cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)
    # find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]
    questionCnts = []

    # loop over the contours
    for c in cnts:
    	# compute the bounding box of the contour, then use the
    	# bounding box to derive the aspect ratio
    	(x, y, w, h) = cv2.boundingRect(c)
    	ar = w / float(h)
    	# in order to label the contour as a question, region
    	# should be sufficiently wide, sufficiently tall, and
    	# have an aspect ratio approximately equal to 1
    	if w >= 10 and h >= 10 and ar >= 0.9 and ar <= 1.1:
    		questionCnts.append(c)
            # sort the question contours top-to-bottom, then initialize
    # the total number of correct answers
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    # each question has 4 possible answers, to loop over the
    # question in batches of 4
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
    	# sort the contours for the current question from
    	# left to right, then initialize the index of the
    	# bubbled answer
    	cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
    	bubbled = None
        temporary = np.zeros_like(thresh)
        cv2.drawContours(temporary, cnts, -1, 255, 5)
        cv2.namedWindow('row_wise_bubble', cv2.WINDOW_NORMAL)
        cv2.imshow('row_wise_bubble', temporary)
        cv2.waitKey(0)
        # loop over the sorted contours
        for (j, c) in enumerate(cnts):
	    	# construct a mask that reveals only the current
	    	# "bubble" for the question
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
	        # apply the mask to the thresholded image, then
	    	# count the number of non-zero pixels in the
	    	# bubble area
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            area = cv2.contourArea(c)
            pix_percentage=float(total)/float(area)
	    	# if the current total has a larger number of total
	    	# non-zero pixels, then we are examining the currently
	    	# bubbled-in answer
            if bubbled is None or pix_percentage > bubbled[0]:
	    		bubbled = (pix_percentage, j+1)

        if bubbled[0] < 0.5:
            bubbled = (bubbled[0], 5)
        bubbled_response.append(bubbled[1])
    return bubbled_response

def main():
    img1 = cv2.imread('omr_scanner.jpg',0)
    img2 = cv2.imread('omr_sheet.jpg',0)

    # Obtain transforms
    trans_img = trans_OMR(img1,img2)
    #
    detl_roi, ans1_roi, ans2_roi = roi_extractor(trans_img)
    bubbled = []
    bubbled = bubble_check(ans1_roi,bubbled)
    bubbled = bubble_check(ans2_roi,bubbled)
    print ("\n\n\nYour Responses: ")
    for i,b in enumerate(bubbled):
        if b == 1:
            print (str(i+1)+":  A")
        elif b == 2:
            print (str(i+1)+":  B")
        elif b == 3:
            print (str(i+1)+":  C")
        elif b == 4:
            print (str(i+1)+":  D")
        else:
            print (str(i+1)+": You have Not Marked Anything")

    (thresh,detl_roi) = cv2.threshold(detl_roi, 128, 255 ,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('test1',detl_roi)
    cv2.waitKey(0)
    tess_in = Image.fromarray(detl_roi)
    print (tess_in)
    tess_in.save('detailed_ROI.jpg')
    text=tesseract.image_to_string(Image.open('detailed_ROI.jpg'))
    print (text)

if __name__=='__main__':
    main()
