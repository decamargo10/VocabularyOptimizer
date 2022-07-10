import random
import os

test_video_list=[{"path": "/home/decamargo/Documents/features_catamaran", "size": 6931, "score":0},
                 {"path": "/home/decamargo/Documents/features", "size": 12000, "score":0},
                 {"path": "/home/decamargo/Documents/features", "size": 12000, "score":0}
                 ]

frame_diffs = [3, 50, 150, 500, -75]

def ev(voc):
    dict = {}
    score = 12
    for i in voc:
        score += int(i)
    return score, dict

def evaluate(voc_path):
    print(voc_path)
    voc_name = os.path.basename(voc_path)
    result_dic = {}
    for video_features in test_video_list:
        correct_detections = 0.0
        attempts = 0.0
        # pick random frame
        for i in range(0,200):
            frame = random.randint(80, video_features["size"]-550)
            frames = []
            for f in frame_diffs:
                frames.append(frame+f)
            print(frames)
            res = is_loop_detected(frame, frames, voc_path)
            attempts += 1
            correct_detections += res
        result_dic[video_features["path"]] = correct_detections/attempts
    avg_score = 0.0
    for v in result_dic:
        avg_score += result_dic[v]
    avg_score = avg_score / len(result_dic)

    # write results into file
    f = open(voc_name, "w")
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
    evaluate("/home/decamargo/Documents/output.fbow")

