import json

frame_dict  = {}
with open('datasample1.json') as json_data:
        d = json.load(json_data)
        for item in d:
            label = d[item]["label"]
            for frame in d[item]["boxes"]:
                if not frame in frame_dict:
                    frame_dict[frame] = []
                print frame, label
                temp_list = [d[item]["boxes"][frame]["xtl"], d[item]["boxes"][frame]["ytl"], d[item]["boxes"][frame]["xbr"], d[item]["boxes"][frame]["ybr"], label]
                frame_dict[frame].append(temp_list)

print frame_dict
