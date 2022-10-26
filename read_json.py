import json
from glob import glob
# Opening JSON file
def read_json_file(file):
    f = open(file)
    data = json.load(f) 
    d = data['annotations']   
    #dic["number_labels"] = len(d)   
    if len(d) == 2:
        return  d[0]["label"], int(d[0]["metadata"]["system"]["frame"]),int(d[0]["metadata"]["system"]["endFrame"]), d[1]["label"], d[1]["metadata"]["system"]["frame"],d[1]["metadata"]["system"]["endFrame"],0,0,0
    elif len(d) == 3:
        return  d[0]["label"], int(d[0]["metadata"]["system"]["frame"]),int(d[0]["metadata"]["system"]["endFrame"]), d[1]["label"], d[1]["metadata"]["system"]["frame"],d[1]["metadata"]["system"]["endFrame"], d[2]["label"], d[2]["metadata"]["system"]["frame"],d[2]["metadata"]["system"]["endFrame"]

    else:
        #print(d)
        for x in d:
            return x["label"], int(x["metadata"]["system"]["frame"]),int(x["metadata"]["system"]["endFrame"]),0,0,0,0,0,0
    #return dic2,names
    f.close()

def read_json_file2(file):
    f = open(file)
    data = json.load(f) 
    d = data['annotations']    
    length = len(d)
    #print(data['filename'])
    print(data["metadata"]["system"]["originalname"])
    print(data["metadata"]["system"]["ffmpeg"]["nb_read_frames"])
    if len(d) == 2: 
        return  d[0]["label"], int(d[0]["metadata"]["system"]["frame"]), int(d[0]["metadata"]["system"]["frame"]+ 10), d[1]["label"], d[1]["metadata"]["system"]["frame"],int(d[1]["metadata"]["system"]["frame"]+ 10),0,0,0,0,0,0
    elif len(d) == 3:
        return  d[0]["label"], int(d[0]["metadata"]["system"]["frame"]), int(d[0]["metadata"]["system"]["frame"]+ 10),d[1]["label"], d[1]["metadata"]["system"]["frame"], int(d[1]["metadata"]["system"]["frame"]+ 10), d[2]["label"], d[2]["metadata"]["system"]["frame"],int(d[2]["metadata"]["system"]["frame"]+ 10),0,0,0
    elif len(d) == 3:
        return  d[0]["label"], int(d[0]["metadata"]["system"]["frame"]),int(d[0]["metadata"]["system"]["frame"]+ 10), d[1]["label"], d[1]["metadata"]["system"]["frame"],int(d[1]["metadata"]["system"]["frame"]+ 10), d[2]["label"], d[2]["metadata"]["system"]["frame"],int(d[2]["metadata"]["system"]["frame"]+ 10),d[3]["label"], d[3]["metadata"]["system"]["frame"],int(d[3]["metadata"]["system"]["frame"]+ 10)
    else:
        for x in d:
            return x["label"], int(x["metadata"]["system"]["frame"]),int(x["metadata"]["system"]["frame"]+ 10),0,0,0,0,0,0,0,0,0
    f.close()



def annotations_dict(folder_labels):
    dict = {}
    for file in folder_labels:
    # dic2,names = read_json_file(file)
        label,start,end,label2,start2,end2,label3,start3,end3  = read_json_file(file)
        file = file.split("\\")[1].split(".")[0]
        dict[file] = label,start,end,label2,start2,end2,label3,start3,end3
    return dict
    #list.append(dic)
    #namess = namess +names

#read_json_file2("json/first_annotations/freekick3.json")
#print(dict['annotations/upload2\\shot9.json'])
#annotations_folder2 = glob("annotations_test/new pens/*")
#d = annotations_dict(annotations_folder2)
#print(d)
