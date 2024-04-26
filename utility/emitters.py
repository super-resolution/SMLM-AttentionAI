import json
import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from third_party.dme import dme
from utility.utility import get_reconstruct_coords, read_thunderstorm_drift_json
from sklearn.neighbors import KDTree

class Emitter():
    """
    Emitter set for SMLM data
    """

    ATTRIBUTES = ["xyz", "photons", "sigxsigy", "frames", "ids", "p"]
    def __init__(self, xyz, photons,  frames,sigxsigy=None, p=None, ids=None):
        self.xyz = np.array(xyz)#in nm
        self.photons = np.array(photons)
        self.sigxsigy = np.array(sigxsigy)#in nm?
        self.frames = np.array(frames,dtype=np.int32)
        self._error = None
        #todo: compute cramer rao lower bound
        if np.any(sigxsigy):
            self.sigxsigy = np.array(sigxsigy)
        else:
            self.sigxsigy = np.zeros((self.xyz.shape[0],2))
        if np.any(ids):
            self.ids = np.array(ids,dtype=np.int32)
        else:
            self.ids = np.arange(0, self.xyz.shape[0])
        if np.any(p):
            self.p = np.array(p)
        else:
            self.p = np.ones(self.xyz.shape[0])
        self.check_data_integrety()
        self.metadata = ""

    def check_data_integrety(self):
        for attr1 in self.ATTRIBUTES:
            for attr2 in self.ATTRIBUTES:
                if getattr(self, attr1).shape[0] != getattr(self,attr2).shape[0] and getattr(self,attr2).shape[0] !=0 and getattr(self, attr1).shape[0] !=0:
                    warnings.warn(f"{attr1} and {attr2} dont have the some length data might be corrupted")


    def add_emitters(self, xyz, photons, sigxsigy, frames, ids):
        """
        Add emitters to current set
        """
        #todo use numpy append here
        self.xyz = np.append(self.xyz, xyz, axis=0)
        self.sigxsigy = np.append(self.sigxsigy, sigxsigy, axis=0)
        self.frames = np.append(self.frames, frames, axis=0)
        self.ids = np.append(self.ids, np.arange(self.ids[-1], xyz.shape[0], 1), axis=0)
        self.check_data_integrety()

    def compute_jaccard2(self, other, images=None):
        #works
        import matplotlib.pyplot as plt
        tp = 0
        fp = 0
        fn = 0
        dis = 0
        offsets = []
        crlb = []
        crlb_per_frame = []
        distances = []
        perc_crlb = 0
        print(crlb)
        other = other.filter(photons=.3)
        for i in range(self.frames.max()//50):
            pred_query = (self.frames>=i*50) & (self.frames<(i+1)*50)& np.all((self.xyz<5900),axis=1) & np.all((self.xyz>100),axis=1)
            points = self.xyz[np.where(pred_query)]
            gt_query = (other.frames>=i*50) & (other.frames<(i+1)*50) & np.all((other.xyz<5900),axis=1) & np.all((other.xyz>100),axis=1)

            othe,ind_ph = np.unique(other.xyz[np.where(gt_query)],axis=0, return_index=True)
            #sum photon trace per emitter
            ids = other.ids[np.where(gt_query)]
            ph = np.zeros_like(ind_ph)
            crlb_per_frame += list(np.sqrt(16/9*(186**2+100**2/12)/other.photons[np.where(gt_query)].astype(np.int32)))
            for j in range(ind_ph.shape[0]):
                ph[j] = np.sum(other.photons[np.where((other.ids==ids[ind_ph[j]])&gt_query)])
            if othe.shape[0]>0:
                tree = KDTree(othe)
                if points.shape[0]>0:
                    #only return closest nearest neighbor
                    x = np.array(tree.query(points))[:,:,0].T
                    #take closest points first
                    y = np.array(sorted(x, key=lambda x: (x[1],x[0])))
                    #condition one find unique neighbor
                    condition1 = np.unique(y[:,1].astype(np.int32),return_index=True)[1]
                    #x = y[indices,:].T
                    offsets += list(points-othe[x[:,1].astype(np.int)])
                    #indices = x[1]
                    crlb += list(np.sqrt(16/9*(186**2+100**2/12)/ph[x[condition1,1].astype(np.int32)]))
                    #perc_crlb += (1*crlb>x[indices,0]).sum()
                    #condition 2 distance <100 nm
                    condition2 = np.where(y[condition1,0]<100)[0]
                    z = y[condition1,0]
                    distances += list(z[condition2])

                    tp += condition2.shape[0]
                    #no loc in the range of 100nm
                    fn += max(othe.shape[0]-condition2.shape[0],0)
                    #distances>100 nm
                    fp += points.shape[0]-np.where(x[:,0]<100)[0].shape[0]
                    # if max(othe.shape[0]-indices.shape[0],0)>0:
                    #     if np.any(images):
                    #         plt.imshow(np.sum(images[i*50:(i+1)*50],axis=0).T)
                    #     plt.scatter(points[:,0]/100,points[:,1]/100, marker="X", label="fit")
                    #     ind = set(range(othe.shape[0]))-set(y[indices,1].astype(np.int32))
                    #     plt.scatter(othe[list(ind),0]/100,othe[list(ind),1]/100, marker="X")
                    #     plt.legend()
                    #     plt.show()
            else:
                fp += points.shape[0]
        print(np.mean(offsets,axis=0))
        plt.hist(crlb, bins=np.arange(0, 50, .5),density=True, label="crlb")
        plt.hist(crlb_per_frame, bins=np.arange(0, 50, .5),density=True, label="crlb_per_frame")
        plt.hist(distances, bins=np.arange(0, 50, .5), alpha=.5,density=True, label="distance")
        plt.ylabel("a.u.")
        plt.xlabel("distance [nm]")
        plt.legend()
        plt.savefig(r"figures/crlb_base_decode.svg")
        plt.show()
        #todo: crlb histogram plot
        print(perc_crlb/tp)
        print("check for offsets", np.mean(np.array(offsets),axis=0))
        print(f"% under crlb {perc_crlb/tp:.2f}")
        #print("check for offsets", np.mean(np.array(offsets),axis=0))
        print(f"TP: {tp}, FN: {fn}, FP{fp}")
        #root mean squared error
        x = np.array(distances)
        print(f"RMSE: {np.sqrt(np.sum(np.array(distances)**2)/tp):.2f} nm")
        print(f"JI: {tp/(fn+tp+fp):.4f}")

    def compute_jaccard(self, other, output="tmp", images=None):
        import matplotlib.pyplot as plt
        tp = 0
        fp = 0
        fn = 0
        dis = 0
        offsets = []
        crlb = np.mean(150/np.sqrt(other.photons))
        perc_crlb = 0
        print(crlb)
        for i in range(self.frames.max()):
            #points = self.xyz[np.where((self.frames>=i*50) & (self.frames<(i+1)*50))]
            #othe = other.xyz[np.where((other.frames>=i*50) & (other.frames<(i+1)*50))]
            points = self.xyz[np.where(self.frames==i)]#-np.array([0,10])[None,:]
            othe = other.xyz[np.where(other.frames==i)]
            ph = other.photons[np.where(other.frames==i)]
            #sigma / sqrt n
            # if i>200:
            #     if np.any(images):
            #         plt.imshow(images[i].T)
            #     plt.scatter(points[:,0]/100,points[:,1]/100, marker="X")
            #     plt.scatter(othe[:,0]/100,othe[:,1]/100, marker="X")
            #     plt.show()
            if othe.shape[0]>0:
                tree = KDTree(othe)
                if points.shape[0]>0:
                    #only return closest nearest neighbor
                    x = np.array(tree.query(points))[:,:,0].T
                    y = np.array(sorted(x, key=lambda x: (x[1],x[0])))
                    indices = np.unique(y[:,1].astype(np.int32),return_index=True)[1]
                    x = y[indices,:].T
                    offsets.append(np.mean(points[indices]-othe[x[1].astype(np.int)],axis=0))
                    x = x[:,np.where(x[0]<100)[0]]#todo: this throws an error
                    indices = x[1]
                    crlb = 180/np.sqrt(ph[x[1].astype(np.int32)])
                    perc_crlb += (2*crlb>x[0]).sum()
                    #todo: select closest
                    dis += np.sum(x[0]**2)
                    tp += indices.shape[0]
                    fn += max(othe.shape[0]-indices.shape[0],0)
                    fp += points.shape[0]-indices.shape[0]
                    # if othe.shape[0]-indices.shape[0]>0:
                    #     #if i>200:
                    #     print(i)
                    #     if np.any(images):
                    #         plt.imshow(images[i].T)
                    #     plt.scatter(points[:,0]/100,points[:,1]/100, marker="X", label="fit")
                    #     plt.scatter(othe[:,0]/100,othe[:,1]/100, marker="X")
                    #     plt.legend()
                    #     plt.show()
            else:
                fp += points.shape[0]
        print("offsets are ", np.mean(np.array(offsets),axis=0))
        with open(f'{output}.txt', 'a') as the_file:
            the_file.write('Hello\n')
            the_file.write(f"% under crlb {perc_crlb/tp:.2f}\n")
            #print("check for offsets", np.mean(np.array(offsets),axis=0))
            the_file.write(f"TP: {tp}, FN: {fn}, FP{fp}\n")
            #root mean squared error
            the_file.write(f"RMSE: {np.sqrt(dis/tp):.2f} nm\n")
            the_file.write(f"JI: {tp/(fn+tp+fp):.4f}\n")
        return np.sqrt(dis/tp),tp/(fn+tp+fp)


    def __add__(self, other):
        """
        Concatenates two emitter set to a new one
        :param other: Emitter set
        """
        self.xyz = np.append(self.xyz, other.xyz, axis=0)
        self.sigxsigy = np.append(self.sigxsigy, other.sigxsigy, axis=0)
        self.frames = np.append(self.frames, self.frames.max()+other.frames+1, axis=0)
        self.ids = np.append(self.ids, other.ids, axis=0)
        self.photons = np.append(self.photons, other.photons, axis=0)
        self.p = np.append(self.p, other.p, axis=0)


    def __mod__(self, other):
        #todo: deprecated
        found_emitters = []
        distances = []
        t= []
        for i in range(int(self.frames.max())):
            pred_f = self.xyz[np.where(self.frames==i)]
            truth_f = other.xyz[np.where(other.frames==i)]
            if np.any(self.ids[np.where(self.frames==i)]):
                id_f = self.ids[np.where(self.frames==i)][0]
                t_truth = []
                for k in range(pred_f.shape[0]):
                    emitter_id = id_f + k
                    found = False
                    min_dis = 100
                    for l in range(truth_f.shape[0]):
                        d = pred_f[k] - truth_f[l]
                        t.append(d)
                        dis = np.linalg.norm(pred_f[k] - truth_f[l])
                        if dis < min_dis:
                            if emitter_id not in found_emitters and l not in t_truth:
                                min_dis = dis
                                current_diff = pred_f[k] - truth_f[l]
                                current_l = l
                                found = True
                    if found:
                        #todo append error to new emitter set?
                        found_emitters.append(emitter_id)
                        t_truth.append(current_l)
                        distances.append(min_dis)

        diff = np.array(found_emitters,dtype=np.int32)
        sub = self.subset(diff)
        sub.error = distances
        return sub
    #compute difference between emitter sets and return emitter set
    def __sub__(self, other):
        """
        Compute the difference between two emitter sets.
        :param other: Emitter set
        :return: Emitters that don`t overlap within a 100nm range
        """
        #todo: deprecated
        found_emitters = []
        distances = []
        for i in range(int(self.frames.max())):
            pred_f = self.xyz[np.where(self.frames==i)]
            truth_f = other.xyz[np.where(other.frames==i)]
            if np.any(self.ids[np.where(self.frames==i)]):
                #todo: ids are not unique anymore
                id_f = np.where(self.frames==i)[0][0]
                t_truth = []
                for k in range(pred_f.shape[0]):
                    emitter_id = id_f + k
                    found = False
                    min_dis = 100
                    for l in range(truth_f.shape[0]):
                        dis = np.linalg.norm(pred_f[k] - truth_f[l])
                        if dis < min_dis:
                            if emitter_id not in found_emitters and l not in t_truth:
                                min_dis = dis
                                current_diff = pred_f[k] - truth_f[l]
                                current_l = l
                                found = True
                                distances.append(min_dis)
                    if found:
                        #todo append error to new emitter set?
                        found_emitters.append(emitter_id)
                        t_truth.append(current_l)
        diff = np.setdiff1d(self.ids, np.array(found_emitters,dtype=np.int32))
        sub = self.subset(diff)
        #sub.error = distances
        return sub

    def adjust_offset(self, offset):
        if isinstance(offset, np.ndarray):
            if offset.shape[0] ==2:
                self.xyz += offset
            else:
                print("wrong shape")
        else:
            print("not an array")

    def subset(self, ids):
        """
        Returns a new emitter set from a list of given ids
        """
        new = Emitter(deepcopy(self.xyz[ids]), deepcopy(self.photons[ids]), deepcopy(self.frames[ids]), deepcopy(self.sigxsigy[ids]),p=deepcopy(self.p[ids]), ids=deepcopy(self.ids[ids]))
        return new

    def filter(self, sig_x=3, sig_y=3, photons=0, p=0, frames=None):
        """
        Filter emitter set
        :param sig_x:
        :param sig_y:
        :param photons:
        :param p:
        :param frames:
        :return: New emitter set with filters applied
        """
        metadata = f"""Filters:\n sigma x:{sig_x}\n sigma y:{sig_y}\n photons:{photons}\n  p:{p}\n frames:{frames}"""
        conditions=[]
        conditions.append(self.sigxsigy[:,0]<sig_x)
        conditions.append(self.sigxsigy[:,1]<sig_y)
        conditions.append(self.photons>photons)
        conditions.append(self.p>p)
        if frames:
            conditions.append(self.frames>frames[0])
            conditions.append(self.frames<frames[1])
        indices = np.array(np.where(np.all(np.array(conditions),axis=0))[0])
        #ids are not unique anymore
        s = self.subset(indices)
        s.metadata += metadata
        return s

    def apply_drift(self, path):
        """
        Apply thunderstorm c-spline drift or raw drif in csv format
        :param path: path to drift correct file
        """
        if isinstance(path, np.ndarray):
            drift = path[:,(1,0)]
        elif path.split(".")[-1] == "csv":
            print("drift correction activated")
            drift = pd.read_csv(path).as_matrix()[::,(1,2)]
            #drift[:,0] *= -1
        else:
            print("drift correction activated")
            path = path+r"\drift.json"
            drift = read_thunderstorm_drift_json(path)
        for i in range(self.frames.max()):
            self.xyz[np.where(self.frames == i),0] += drift[i,1]*100
            self.xyz[np.where(self.frames == i),1] -= drift[i,0]*100

    def use_dme_drift_correct(self):
        use_cuda = True
        fov_width = 80
        loc_error = np.array((0.2, 0.2, 0.03))  # pixel, pixel, um
        loc = self.xyz
        localizations = np.zeros((loc.shape[0], 3))
        localizations[:, 0:2] = loc / 100
        crlb = np.ones(localizations.shape) * np.array(loc_error)[None]
        estimated_drift, _ = dme.dme_estimate(localizations, self.frames,
                                          crlb,
                                          framesperbin=200,  # note that small frames per bin use many more iterations
                                          imgshape=[fov_width, fov_width],
                                          coarseFramesPerBin=200,
                                          coarseSigma=[0.2, 0.2, 0.2],  # run a coarse drift correction with large Z sigma
                                          useCuda=use_cuda,
                                          useDebugLibrary=False)
        self.apply_drift(estimated_drift[:,0:2])


    def save(self, path, format="npy"):
        """
        Save emitter set in the given format
        :param path: save path
        :param format: save format
        :return:
        """
        names = {}
        data = []
        col = 0
        for att in self.ATTRIBUTES:
            val = getattr(self, att)
            if np.any(val):
                if len(val.shape)==1:
                    names[att] = [col, 1]

                    data.append(val[:,None])
                    col += 1
                else:
                    names[att] = [col, val.shape[1]]
                    data.append(val)
                    col+= val.shape[1]
        names["metadata"] = self.metadata
        data = np.concatenate(data, axis=-1)
        if format =="npy":
            #make if exists test for non mandatory columns
            path_m = os.path.splitext(path)
            with open(path_m[0] + '_metadata.json', 'w') as fp:
                json.dump(names, fp, sort_keys=True, indent=4)
            np.save(path, data)
        elif format == "csv":
            pass
        elif format == "txt":
            pass


    @classmethod
    def load(cls, path, raw=True, format="npy"):
        """
        Load emitter set from given path
        :param path:
        :param raw:
        :return:
        """
        # make if exists test for non mandatory columns
        path_m = os.path.splitext(path)
        #todo: get collumn names
        with open(path_m[0] + '_metadata.json') as fp:
            name = json.load(fp)
        data = np.load(path)
        d= {}
        for att in cls.ATTRIBUTES:
            if att in name:
                d[att] = np.squeeze(data[:,name[att][0]:name[att][0]+name[att][1]])
        return cls(**d)



    @classmethod
    def from_result_tensor(cls, result_tensor, p_threshold, coord_list=None, gpu=False,
                           maps={"p":0, "x":1,"y":2,"dx":3,"dy":4,"N":5,"dN":6}):
        """
        Build an emitter set from the neural network output
        :param result_tensor: Feature map output of the AI
        :param coord_list: Coordinates of the crops detected by the wavelet filter bank
        :param p_threshold: Threshold value for a classifier pixel to contain a localisation
        :return:
        """
        if gpu:
            pass
        else:
            if not np.any(coord_list):
                print("warning no coordinate list loaded -> no offsets are added")

            xyz = []
            N_list = []
            sigx_sigy = []
            sN_list = []
            p_list = []
            frames = []
            for i in range(result_tensor.shape[0]):
                classifier =result_tensor[i,maps["p"], :, :]
                #if i==32:
                #    x=0
                x= np.sum(classifier)
                # plt.imshow(classifier)
                # plt.show()
                if x > p_threshold:
                    #todo: get indices and p from function...
                    indices,p = get_reconstruct_coords(classifier, p_threshold)#todo: import and use function

                    x = result_tensor[i,maps["x"], indices[0], indices[1]]
                    y = result_tensor[i,maps["y"], indices[0], indices[1]]
                    dx = result_tensor[i,maps["dx"], indices[0], indices[1]]
                    dy = result_tensor[i,maps["dx"], indices[0], indices[1]]
                    N = result_tensor[i,maps["N"], indices[0], indices[1]]
                    dN = result_tensor[i,maps["dN"], indices[0], indices[1]]

                    for j in range(indices[0].shape[0]):
                        if np.any(coord_list):
                            xyz.append(np.array([coord_list[i][0] + 0.5 + float(indices[0][j]) + (x[j]),#x
                                                 coord_list[i][1] + 0.5 +float(indices[1][j]) + y[j]])*100)#y
                            frames.append(coord_list[i][2])
                        else:
                            xyz.append(np.array([float(indices[0][j])+ 0.5 + (x[j]),#x
                                                 float(indices[1][j])+ 0.5 + y[j]])*100)#y
                            frames.append(i)
                        p_list.append(p[j])
                        sigx_sigy.append(np.array([dx[j],dy[j]]))
                        N_list.append(N[j])
                        sN_list.append(dN[j])
        return cls(xyz, N_list, frames, sigx_sigy, p_list)#+50 for center pix offset

    @classmethod
    def from_ground_truth(cls, coordinate_tensor):
        coords = []
        frames = []
        photons = []
        ids = []
        for i, crop in enumerate(coordinate_tensor):
            for coord in crop:
                #if coord[2] != 0:
                    #todo add photons
                if coord[0] != 0:
                    coords.append(np.array([coord[0], coord[1], i, coord[2]]))#x,y,frame,intensity
                    frames.append(i)
                    photons.append(coord[2])
                    ids.append(coord[4])
        coords = np.array(coords)
        coords[:, 0:2] *= 100
        return cls(coords[:,0:2], np.array(photons), np.array(frames), ids=np.array(ids))

    @classmethod
    def from_thunderstorm_csv(cls, path, contest=False):
        results = pd.read_csv(path).to_numpy()
        if contest:
            frame = results[:,1]-1
            xyz = results[:,2:4]-50
            I = results[:,5]
        else:
            frame = results[:,0]
            xyz = results[:,1:3]-50
            I = results[:,4]

        return cls(xyz[:,::-1], I, frame)


    def get_frameset_generator(self, images, frames, gt=None):
        """
        Generator for plotting emitter sets on images
        :param images:
        :param gt:
        :param frames:
        :return: Generator with image, emitters, ground truth
        """
        def frameset_generator():
            for frame in frames:
                emitters = self.xyz[np.where(self.frames==frame)]/100
                if np.any(emitters):
                    print(f"current frame: {frame}")
                    image = images[frame]
                    if gt:
                        gt_emitters = gt.xyz[np.where(gt.frames==frame)]/100
                        yield image[:,:,1], emitters, gt_emitters
                    else:
                        yield image, emitters
        return frameset_generator

    @property
    def error(self):
        """returns RMSE"""
        if np.any(self._error):
            return np.sqrt(np.mean(self._error**2))
        else:
            print("no error available")

    @error.setter
    def error(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        if value.shape[0] == self.ids.shape[0]:
            self._error = value
        else:
            print("critical error error has an error length")


    @property
    def length(self):
        return self.ids.shape[0]