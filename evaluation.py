import random
import os
import sys

sys.path.insert(0, '/home/decamargo/Documents/FBoW/build')
import fbow_pybind

test_video_list=[{"path": "/home/decamargo/Documents/features_catamaran", "size": 6931, "score":0},
                 {"path": "/home/decamargo/Documents/features_ice_breaker", "size": 5543, "score":0},
                 {"path": "/home/decamargo/Documents/features_ships_horiz1", "size": 4176, "score":0},
                 {"path": "/home/decamargo/Documents/features_ships_horiz2", "size": 5851, "score":0},
                 {"path": "/home/decamargo/Documents/features_ships_horiz3", "size": 6416, "score":0},
                 {"path": "/home/decamargo/Documents/features_ship_uu", "size": 4499, "score":0}
                 ]

frame_diffs = [3, 80, 175, 500, -85]

def ev(voc):
    dict = {}
    score = 12
    tests = []
    test = [1,2,3,4,5]
    tests.append(test)
    test = [1,30,13,111,4]
    tests.append(test)
    res = fbow_pybind.detect_loops(tests, )
    print(res)
    return score, dict

def evaluate(voc_path, norm, weight):
    print(voc_path)
    voc_name = os.path.basename(voc_path)
    result_dic = {}
    for video_features in test_video_list:
        attempts = 0.0
        # pick random frame
        tests=[]
        for i in range(0,200):
            test = []
            frame = random.randint(100, video_features["size"]-550)
            test.append(frame)
            for f in frame_diffs:
                test.append(frame+f)
            attempts += 1
            tests.append(test)
        use_tf = False
        if weight == 1 or weight == 3:
            use_tf=True
        correct_detections = fbow_pybind.detect_loops(tests, video_features["path"], voc_path, norm-1, use_tf)
        result_dic[video_features["path"]] = correct_detections/attempts
    avg_score = 0.0
    for v in result_dic:
        avg_score += result_dic[v]
    avg_score = avg_score / len(result_dic)
    file_name = "./reports/" + voc_name + ".txt"
    # write results into file
    f = open(file_name, "w")
    f.write("Results for vocabulary: " + voc_name + "\n____________________\n")
    f.write("Average score over all test sets: " + str(avg_score) + "\n")
    for k in result_dic:
        f.write(k + ": " + str(result_dic[k]) + "\n")
    f.close()
    return avg_score, result_dic


def is_loop_detected(frame, frames, voc_path):
    loop_score = get_score(frame, frames[0])
    other_scores = []
    for f in frames:
        s = get_score(frame, f)
        other_scores.append(s)
    for s in other_scores:
        if s >= loop_score:
            return 0
    return 1


def get_score():
    return 0


if __name__ == '__main__':
    # we get a freshly generated voc and evaluate it here
    #evaluate("/home/decamargo/Downloads/orb_vocab.fbow",2, 3)
    evaluate("/home/decamargo/PycharmProjects/ba_evaluation/reports_pop20_gen257/100101111100.voc", 0, 0)
    #ev("test")

