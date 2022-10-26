import cv2
def get_label(names_test,video,pred,y_test,frame,seq):
    sample = frame//seq
    if video.split("\\")[1].split(".")[0] != names_test[sample]:
        return "n","stop"  
    elif pred[sample] == 0:
        print(sample,video,names_test[sample],pred[sample],y_test[sample]) 
        return "No-Penalty","right"
    elif pred[sample] == 1:
        print(sample,video,names_test[sample],pred[sample],y_test[sample]) 
        return "Penalty","right"

def play_video(names_test,video,pred,y_test,n_samples,seq):  #play each frame with corresponding label
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    window_name = "window"
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 840, 480)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    count = 0
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')#1920,1080 640.
    #out = cv2.VideoWriter(video+".mp4", fourcc, fps, (frame_width,frame_height))
    pos = 0
    neg = 0
    mins = 0
    sec = 0
    period = '00:00:00:00'
    while(cap.isOpened()):
        print(frame_width,frame_height)
        text,right = get_label(names_test,video,pred,y_test,n_samples*seq+count,seq)
        ret, frame = cap.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        if ret == True and right == "right":
            cfn = cap.get(1)
            print(cfn,fps,int(cfn)%int(fps))

            #print timer
            if int(cfn)%int(fps)==1:
                if sec > 59:
                    sec = 0
                    mins = mins+1
                period = "{:02d}:{:02d}".format(mins,sec)
                sec = sec + 1
            if frame_width != 1920:
                cv2.putText(frame,period, (2*1500,2*1000),font,6,(0,0,0),30)
            else:
             cv2.putText(frame,period, (1500,1000),font,3,(0,0,0),15)

            #print labels
            if text == "No-Penalty":
                if frame_width != 1920:
                    cv2.putText(frame,text, (2*0,2*150),font,10,(0,255,0),30)
                else:
                    cv2.putText(frame,text, (0,150),font,5,(0,255,0),15)
                neg = neg + 1
            elif text == "Penalty":
                pos = pos + 1
                if frame_width != 1920:
                    cv2.putText(frame,text, (2*0,2*150),font,10,(0,0,255),30)
                else:
                    cv2.putText(frame,text, (0,150),font,5,(0,0,255),15)

           #out.write(frame)
            cv2.imshow(window_name, frame)
            count = count + 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Pen:",pos,"No pen:",neg)
                break
        else:
            print("Pen:",pos,"No pen:",neg)
            return n_samples+count
            break
    
    # release the cap object
    cap.release()
    # close all windows
    cv2.destroyAllWindows()

