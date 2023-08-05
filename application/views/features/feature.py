import os
import numpy as np
import matplotlib.pyplot as plt

from time import time
from application.views.database_utils.data import Data
from sklearn.manifold import TSNE

class VideoFeature():
    def __init__(self, data: Data, mode="pred", reload=False, retsne=False):
        self.path = "application/views/features/" + data.dataset + "/"
        if not os.path.exists(self.path) or not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.data = data
        self.frames = []

        # dataset config:
        self.back_index = 0
        self.cls = [[]]
        self.fps = 25
        # ---thumos
        if data.dataset == "thumos":
            self.back_index = 20
            self.cls = [[5, 6, 17], [0, 1, 16, 19], [11, 13, 14], [8, 10, 12, 15, 18], [4, 7], [2], [9], [3]]
        # ---thumos19
        if data.dataset == "thumos19":
            self.back_index = 19
            self.cls = [[4, 5, 16], [0, 1, 15, 18], [10, 12, 13], [7, 9, 11, 14, 17], [6], [2], [8], [3]]

        self.mode = mode            # anns 、pred
        self.reload = reload        # reload features
        self.retsne = retsne   # is tsne finished
        self.draw_tsne = False      # do tsne painting in backend

    def read_tsne(self):
        """Read tsne file
        :return:
        """
        if not self.retsne:
            result = np.load(self.path + self.mode + "_tsne_data.npy")
            label = np.load(self.path + self.mode + "_label.npy")
        else:
            result, label = self.tsne()
            result[:, 0] = (result[:, 0] - np.min(result[:, 0])) / \
                           (np.max(result[:, 0]) - np.min(result[:, 0]))
            result[:, 1] = (result[:, 1] - np.min(result[:, 1])) / \
                           (np.max(result[:, 1]) - np.min(result[:, 1]))
            np.save(self.path + self.mode + "_tsne_data.npy", result)
        return result, label

    def tsne(self):
        """Extract and do tsne
        :return:
        """
        if self.reload:
            data, label = self.extract_features()
        else:
            data = np.load(self.path + self.mode + "_data.npy")
            label = np.load(self.path + self.mode + "_label.npy")
        return self.tsne_on_features(data, label)

    def extract_video_action_features(self, vid: int):
        """Extract video action features and background features

        :param vid:int id of target video
        :return action_feature:dict action index and features
                back_feature:dict  index and features
        """
        assert vid in self.data.meta_data["train_idx"]

        action_feature, back_feature = {}, {}

        # generate anns
        print("Extract action feature: video " + str(vid))
        num_frame = self.data.meta_data["num_frames"][vid]
        annotated_frames = self.data.meta_data["annotated_frames"][vid]
        anns = [self.back_index] * num_frame
        for index in annotated_frames:
            anns[index] = self.data.get_ground_truth_of_single_frame(vid, index)

        # find action features
        act_list = []
        act_feas = []
        t0 = time()
        for index in annotated_frames:
            action = self.data.get_ground_truth_of_single_frame(vid, index)
            feature = self.data.get_feature_of_single_frame(vid, index)
            count = 1

            # search in left aspect
            i = 1
            while index - i >= 0 and anns[index - i] == self.back_index:
                if self.mode == "pred":
                    clas = self.data.get_classification_of_single_frame(vid, index - i)
                    if action in np.where(clas == np.max(clas)):
                        feature += self.data.get_feature_of_single_frame(vid, index - i)
                        anns[index - i] = action
                        count += 1
                    else:
                        break
                    i += 1
                else:
                    clas = self.data.get_ground_truth_of_single_frame(vid, index - i)
                    if clas == action:
                        feature += self.data.get_feature_of_single_frame(vid, index - i)
                        anns[index - i] = action
                        count += 1
                    else:
                        break
                    i += 1
            left = index - i + 1
            # search in right aspect
            i = 1
            while index + i < num_frame and anns[index + i] == self.back_index:
                if self.mode == "pred":
                    clas = self.data.get_classification_of_single_frame(vid, index + i)
                    if action in np.where(clas == np.max(clas)):
                        feature += self.data.get_feature_of_single_frame(vid, index + i)
                        anns[index + i] = action
                        count += 1
                    else:
                        break
                    i += 1
                else:
                    clas = self.data.get_ground_truth_of_single_frame(vid, index + i)
                    if clas == action:
                        feature += self.data.get_feature_of_single_frame(vid, index + i)
                        anns[index + i] = action
                        count += 1
                    else:
                        break
                    i += 1
            right = index + i - 1
            act_list.append(action)
            feature = feature / count
            act_feas.append(feature)
            print(left / self.fps,  right / self.fps, index / self.fps)

        action_feature["action"] = act_list
        action_feature["index"] = annotated_frames
        action_feature["feature"] = act_feas
        print("\tAction:\t" + str(round(time() - t0, 2)) + "s")

        # find back features
        t0 = time()
        back_list = []
        back_index = []
        back_feas = []
        start, in_back = 0, False
        feature, count = [], 0
        for index in range(num_frame):
            if not in_back and anns[index] == self.back_index:
                start = index
                in_back = True
                feature = self.data.get_feature_of_single_frame(vid, index)
                count = 1
            elif in_back:
                if anns[index] == self.back_index:
                    feature += self.data.get_feature_of_single_frame(vid, index)
                    count += 1
                else:
                    in_back = False
                    feature = feature / count
                    back_list.append(self.back_index)
                    back_index.append([start, index-1])
                    back_feas.append(feature)
        if in_back:
            feature = feature / count
            back_list.append(self.back_index)
            back_index.append([start, num_frame-1])
            back_feas.append(feature)

        back_feature["action"] = back_list
        back_feature["index"] = back_index
        back_feature["feature"] = back_feas
        print("\tBackground:\t" + str(round(time() - t0, 2)) + "s\n")

        return action_feature, back_feature

    def extract_features(self):
        """Extract all features

        :return data:ndarray features
                label:list labels
        """
        data, label = [], []
        for vid in self.data.meta_data["train_idx"]:
            acs, bks = self.extract_video_action_features(vid)
            data.extend(acs["feature"])
            label.extend(acs["action"])
            data.extend(bks["feature"])
            label.extend(bks["action"])

        print(len(label))
        data = np.array(data)
        label = np.array(label)
        np.save(self.path + self.mode + "_data.npy", data)
        np.save(self.path + self.mode + "_label.npy", label)
        return data, label

    def tsne_on_features(self, data, label):
        """TSNE and draw plot

        :param data:ndarray, label:list
        :return
        """
        '''sklearn tsne
        tsne = TSNE(n_components=2, init="pca", learning_rate=100)
        result = tsne.fit_transform(data)
        '''
        tsne = IncrementalTSNE(n_components=2, init='pca', method='barnes_hut', perplexity=30, angle=0.3, n_jobs=8,
                               n_iter=1000)
        result = tsne.fit_transform(data, labels=label, label_alpha=0.6)
        if self.draw_tsne:
            t0 = time()
            self._plot_embedding(result, label,
                             't-SNE embedding of the digits (time %.2fs)'
                             % (time() - t0))
        return result, label

    def _plot_embedding(self, data, label, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        colors = ['silver'] * 21
        sports = self.cls
        colormap = ["pink", "red", "yellow", "green", "blue", "cyan", "purple", "salmon"]
        for i in range(8):
            for j in sports[i]:
                colors[j] = colormap[i]

        ax = plt.subplot(111)
        for i in range(data.shape[0]):
            if label[i] == 20:
                continue
            plt.scatter(data[i, 0], data[i, 1],
                     color=colors[label[i]])
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        plt.show()

    def test_speed(self, vid: int):
        """Test query speed
        :param vid: video id
        :return:
        """
        test_vid = vid
        num_frame = self.data.meta_data["num_frames"][test_vid]
        print("Test query speed: video " + str(vid) + ", total frames: " + str(num_frame))

        t0 = time()
        for i in range(num_frame):
            self.data.get_classification_of_single_frame(test_vid, i)
        t1 = time()
        for i in range(num_frame):
            self.data.get_feature_of_single_frame(test_vid, i)
        t2 = time()
        for i in range(num_frame):
            self.data.get_action_score_of_single_frame(test_vid, i)
        t3 = time()

        print("\tcls:\t" + str((t1 - t0) / num_frame) + "s")
        print("\tfea:\t" + str((t2 - t1) / num_frame) + "s")
        print("\tact:\t" + str((t3 - t2) / num_frame) + "s")