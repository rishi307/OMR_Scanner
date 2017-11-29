import cv2
import numpy as np 
def roi_extractor(img)
    ret,img_inv=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    print img_inv
    im2,cnts,hier=cv2.findContours(img_inv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    area=[]
    for i,cnt in enumerate(cnts):
        area.append(cv2.contourArea(cnt))
    idx=np.argsort(area)
    detail=cnts[idx[-1]]
    ans1=cnts[idx[-2]]
    ans2=cnts[idx[-3]]
    dc=cv2.boundingRect(detail)
    ac1=cv2.boundingRect(ans1)
    ac2=cv2.boundingRect(ans2)
    detl_roi=img[dc[1]:dc[1]+dc[3],dc[0]:dc[0]+dc[2]]
    detl_roi=img[dc[1]:dc[1]+dc[3],dc[0]:dc[0]+dc[2]]
    if ac1[0]>ac2[0]:
        ans1_roi=img[ac2[1]:ac2[1]+ac2[3],ac2[0]:ac2[0]+ac2[2]]
        ans2_roi=img[ac1[1]:ac1[1]+ac1[3],ac1[0]:ac1[0]+ac1[2]]
    else:
        ans1_roi=img[ac1[1]:ac1[1]+ac1[3],ac1[0]:ac1[0]+ac1[2]]
        ans2_roi=img[ac2[1]:ac2[1]+ac2[3],ac2[0]:ac2[0]+ac2[2]]

return detl_roi,ans1_roi,ans2_roi