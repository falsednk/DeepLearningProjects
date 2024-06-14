import os
import sys
from tqdm import tqdm
from pathlib import Path
from fastprogress import progress_bar

import numpy as np
import h5py
from copy import deepcopy
from typing import Any
import itertools
from collections import defaultdict
import cv2
import torch
from torch import Tensor as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import kornia as K
import kornia.feature as KF
from kornia.feature import LoFTR
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from lightglue import match_pair
from lightglue import LightGlue, ALIKED, SuperPoint, DISK
from lightglue.utils import load_image, rbd

from dkm.utils.utils import tensor_to_pil, get_tuple_transform_ops
from dkm import DKMv3_outdoor


def load_torch_image(file_name: Path | str, device=torch.device('cpu')):
    img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img


def import_into_colmap(path: Path, feature_dir: Path, database_path: str = 'colmap.db') -> None:
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, path, '', 'simple-pinhole', single_camera)
    add_matches(db, feature_dir, fname_to_id)
    db.commit()


class GetPairs:
    def __init__(self, model_name: str, paths: list[Path], device: torch.device = torch.device('cpu')):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(device)
        self.device = device
        self.paths = paths

    def embed_images(self) -> T:
        embeddings = []
        for i, path in tqdm(enumerate(self.paths), desc='Global descriptors'):
            image = load_torch_image(path, device=self.device)
            with torch.inference_mode():
                inputs = self.processor(images=image, return_tensors='pt', do_rescale=False).to(self.device)
                outputs = self.model(**inputs)

            embedding = F.normalize(outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=-1, p=2)
        embeddings.append(embedding.detach().cpu())
        return torch.cat(embeddings, dim=0)

    def get_pairs_exhaustive(self, lst: list[Any]) -> list[tuple[int, int]]:
        return list(itertools.combinations(range(len(lst)), 2))

    def get_image_pairs(self, similarity_threshold: float = 0.6,
                        tolerance: int = 1000, min_matches: int = 20, exhaustive_if_less: int = 20,
                        p: float = 2.0) -> list[tuple[int, int]]:

        if len(self.paths) <= exhaustive_if_less:
            return self.get_pairs_exhaustive(self.paths)
        matches = []

        embeddings = self.embed_images()
        distances = torch.cdist(embeddings, embeddings, p=p)

        mask = distances <= similarity_threshold
        image_indices = np.arange(len(self.paths))

        for current_image_index in range(len(self.paths)):
            mask_row = mask[current_image_index]
            indices_to_match = image_indices[mask_row]

            if len(indices_to_match) < min_matches:
                indices_to_match = np.argsort(distances[current_image_index])[:min_matches]

            for other_image_index in indices_to_match:
                if other_image_index == current_image_index:
                    continue
                if distances[current_image_index, other_image_index] < tolerance:
                    matches.append(tuple(sorted((current_image_index, other_image_index.item()))))

        return sorted(list(set(matches)))


class LightglueReconstruction:
    def __init__(self, model_name: str, paths: list[Path], device: torch.device = torch.device('cpu')):
        self.paths = paths
        self.model_name = model_name
        self.device = device

    def detect_keypoints(self, feature_dir: Path, num_features: int = 4096, resize_to: int = 1024, \
                         detection_threshold: float = 0.01) -> None:

        dtype = torch.float32
        extractors = {'aliked': ALIKED, 'superpoint': SuperPoint, 'disk': DISK}
        extractor = extractors[self.model_name](max_num_keypoints=num_features, detection_threshold=detection_threshold,
                                                resize=resize_to).eval().to(self.device, dtype)
        feature_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(f'{feature_dir}/keypoints_{self.model_name}.h5', mode='w') as f_keypoints, \
                h5py.File(f'{feature_dir}/descriptors_{self.model_name}.h5', mode='w') as f_descriptors:
            for path in tqdm(self.paths, desc='Computing keypoints'):
                key = path.name
                with torch.inference_mode():
                    image = load_torch_image(path, device=self.device).to(dtype)
                    features = extractor.extract(image)
                    f_keypoints[key] = features['keypoints'].reshape(-1, 2).detach().cpu().numpy()
                    f_descriptors[key] = features['descriptors'].reshape(len(f_keypoints[key]),
                                                                         -1).detach().cpu().numpy()

    def keypoint_distances(self, index_pairs: list[tuple[int, int]], file_keypoints,
                           feature_dir: Path, min_matches: int = 15, verbose: bool = True) -> None:

        matcher_params = {'width_confidence': -1, 'depth_confidence': -1,
                          'mp': True if 'cuda' in str(self.device) else False}
        matcher = KF.LightGlueMatcher(self.model_name, matcher_params).eval().to(self.device)

        with h5py.File(f'{feature_dir}/keypoints_{self.model_name}.h5', mode='r') as f_keypoints, \
                h5py.File(f'{feature_dir}/descriptors_{self.model_name}.h5', mode='r') as f_descriptors, \
                h5py.File(file_keypoints, mode='w') as f_matches:
            for idx1, idx2 in tqdm(index_pairs, desc='Computing keypoing distances'):
                key1, key2 = self.paths[idx1].name, self.paths[idx2].name
                keypoints1 = torch.from_numpy(f_keypoints[key1][...]).to(self.device)
                keypoints2 = torch.from_numpy(f_keypoints[key2][...]).to(self.device)
                descriptors1 = torch.from_numpy(f_descriptors[key1][...]).to(self.device)
                descriptors2 = torch.from_numpy(f_descriptors[key2][...]).to(self.device)

                with torch.inference_mode():
                    distances, indices = matcher(descriptors1, descriptors2,
                                                 KF.laf_from_center_scale_ori(keypoints1[None]),
                                                 KF.laf_from_center_scale_ori(keypoints2[None]))

                if len(indices) == 0:
                    continue
                keypoints1 = keypoints1[indices[:, 0], :].cpu().numpy().reshape(-1, 2).astype(np.float32)
                keypoints2 = keypoints2[indices[:, 1], :].cpu().numpy().reshape(-1, 2).astype(np.float32)
                n_matches = len(indices)
                if n_matches:
                    if n_matches >= min_matches:
                        group = f_matches.require_group(key1)
                        group.create_dataset(key2, data=np.concatenate([keypoints1, keypoints2], axis=1))


class DKMDataset(Dataset):
    def __init__(self, file_names_1: list, file_names_2: list, resize_to: int,
                 device: torch.device = torch.device('cpu')):
        self.file_names_1 = file_names_1
        self.file_names_2 = file_names_2
        self.resize_to = resize_to
        self.device = device
        self.test_transform = get_tuple_transform_ops(resize=self.resize_to, normalize=True)

    def __len__(self):
        return len(self.file_names_1)

    def __getitem__(self, idx):
        fname1 = self.file_names_1[idx]
        fname2 = self.file_names_2[idx]

        img1, img2 = Image.open(fname1), Image.open(fname2)
        ori_shape_1, ori_shape_2 = img1.size, img2.size
        image1, image2 = self.test_transform((img1, img2))
        return image1, image2, torch.tensor([idx]), torch.tensor(ori_shape_1), torch.tensor(ori_shape_2)


class DKMReconstruction:
    def __init__(self, paths: list[Path], resize_to: tuple[int, int] = (540, 720), batch_size: int = 4,
                 device: torch.device = torch.device('cpu'), detection_threshold: float = 0.5,
                 num_features: int = 2000, min_matches: int = 15):
        self.paths = paths
        self.resize_to = resize_to
        self.device = device
        self.batch_size = batch_size
        self.model = DKMv3_outdoor(device=self.device)
        self.model.upsample_preds = False
        self.detection_threshold = detection_threshold
        self.num_features = num_features
        self.min_matches = min_matches

    def get_dkm_dataloader(self, file_names_1: list, file_names_2: list):
        dataset = DKMDataset(file_names_1, file_names_2, self.resize_to, self.device)
        dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=self.batch_size,
                                pin_memory=True, num_workers=2, drop_last=False)
        return dataloader

    def get_dkm_mkpts(self, bimgs1, bimgs2, shapes1, shapes2):

        dense_matches, dense_certainty = self.model.match(bimgs1, bimgs2, batched=True)
        store_mkpts1, store_mkpts2, store_mconf = [], [], []

        for b in range(dense_matches.shape[0]):
            u_dense_matches = dense_matches[b, dense_certainty[b, ...].sqrt() >= self.detection_threshold, :]
            u_dense_certainty = dense_certainty[b, dense_certainty[b, ...].sqrt() >= self.detection_threshold]

            if u_dense_matches.shape[0] > self.num_features:
                u_dense_matches, u_dense_certainty = self.model.sample(u_dense_matches, u_dense_certainty,
                                                                       num=self.num_features)

            u_dense_matches = u_dense_matches.reshape((-1, 4))
            u_dense_certainty = u_dense_certainty.reshape((-1,))

            mkpts1 = u_dense_matches[:, :2]
            mkpts2 = u_dense_matches[:, 2:]

            w1, h1 = shapes1[b, :]
            w2, h2 = shapes2[b, :]

            mkpts1[:, 0] = ((mkpts1[:, 0] + 1) / 2) * w1
            mkpts1[:, 1] = ((mkpts1[:, 1] + 1) / 2) * h1

            mkpts2[:, 0] = ((mkpts2[:, 0] + 1) / 2) * w2
            mkpts2[:, 1] = ((mkpts2[:, 1] + 1) / 2) * h2

            mkpts1 = mkpts1.cpu().detach().numpy()
            mkpts2 = mkpts2.cpu().detach().numpy()
            mconf = u_dense_certainty.sqrt().cpu().detach().numpy()

            if mconf.shape[0] > self.min_matches:
                try:
                    F, inliers = cv2.findFundamentalMat(mkpts1, mkpts2, cv2.USAC_MAGSAC, 0.200, 0.999, 2000)
                    inliers = inliers > 0
                    mkpts1 = mkpts1[inliers[:, 0]]
                    mkpts2 = mkpts2[inliers[:, 0]]
                    mconf = mconf[inliers[:, 0]]
                except:
                    pass
            store_mkpts1.append(mkpts1)
            store_mkpts2.append(mkpts2)
            store_mconf.append(mconf)
        return store_mkpts1, store_mkpts2, store_mconf

    def detect_dkm(self, index_pairs: list[tuple[int, int]], feature_dir: Path, file_keypoints) -> None:

        file_names_1, file_names_2 = [], []
        for pair_idx in progress_bar(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = self.paths[idx1], self.paths[idx2]
            file_names_1.append(fname1)
            file_names_2.append(fname2)

        with h5py.File(file_keypoints, mode='w') as f_match:
            dataloader = self.get_dkm_dataloader(file_names_1, file_names_2)
            for X in tqdm(dataloader):
                images1, images2, idxs, shapes1, shapes2 = X
                store_mkpts1, store_mkpts2, store_mconf = self.get_dkm_mkpts(images1.to(self.device), images2.to(self.device),
                                                                             shapes1, shapes2)

            for b in range(images1.shape[0]):
                mkpts1, mkpts2 = store_mkpts1[b], store_mkpts2[b]
                mconf = store_mconf[b]
                file1, file2 = file_names_1[idxs[b]], file_names_2[idxs[b]]
                key1, key2 = file1.name, file2.name

                n_matches = mconf.shape[0]
                group = f_match.require_group(key1)
                if n_matches >= self.min_matches:
                    group.create_dataset(key2, data=np.concatenate([mkpts1, mkpts2], axis=1).astype(np.float32))


class LoFTRDataset(Dataset):
    def __init__(self, file_names_1: list, file_names_2: list, idxs1: list, idxs2: list, resize_small_edge_to: int,
                 device: torch.device = torch.device('cpu')):
        self.file_names_1 = file_names_1
        self.file_names_2 = file_names_2
        self.keys1 = [filename.name for filename in file_names_1]
        self.keys2 = [filename.name for filename in file_names_2]
        self.idxs1 = idxs1
        self.idxs2 = idxs2
        self.resize_small_edge_to = resize_small_edge_to
        self.device = device
        self.round_unit = 16

    def __len__(self):
        return len(self.file_names_1)

    def load_torch_image_(self, filename):
        img = cv2.imread(str(filename))
        original_shape = img.shape
        ratio = self.resize_small_edge_to / min([img.shape[0], img.shape[1]])
        w = int(img.shape[1] * ratio)
        h = int(img.shape[0] * ratio)
        img_resized = cv2.resize(img, (w, h))
        img_resized = K.image_to_tensor(img_resized, False).float() / 255.
        img_resized = K.color.bgr_to_rgb(img_resized)
        img_resized = K.color.rgb_to_grayscale(img_resized)
        return img_resized.to(self.device), original_shape

    def __getitem__(self, idx):
        fname1 = self.file_name_1[idx]
        fname2 = self.file_name_2[idx]
        image1, ori_shape_1 = self.load_torch_image_(fname1)
        image2, ori_shape_2 = self.load_torch_image_(fname2)
        return image1, image2, self.keys1[idx], self.keys2[idx], self.idxs1[idx], self.idxs2[
            idx], ori_shape_1, ori_shape_2


class LoFTRReconstruction:
    def __init__(self, paths: list[Path], path_weights: str, resize_small_edge_to: int = 750, batch_size: int = 4,
                 min_matches: int = 15, device: torch.device = torch.device('cpu')):
        self.paths = paths
        self.resize_small_edge_to = resize_small_edge_to
        self.device = device
        self.batch_size = batch_size
        self.model = LoFTR(pretrained=None)
        self.model.load_state_dict(torch.load(path_weights)['state_dict'])
        self.model = self.model.to(self.device).eval()
        self.min_matches = min_matches

    def get_loftr_dataloader(self, file_names_1: list, file_names_2: list, idxs1: list, idxs2: list):
        dataset = LoFTRDataset(file_names_1, file_names_2, idxs1, idxs2, self.resize_small_edge_to, self.device)
        dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=self.batch_size,
                                pin_memory=True, num_workers=2, drop_last=False)
        return dataset

    def detect_loftr(self, index_pairs: list[tuple[int, int]], feature_dir: Path, file_keypoints) -> None:

        file_names_1, file_names_2, idxs1, idxs2 = [], [], [], []
        for pair_idx in progress_bar(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = self.paths[idx1], self.paths[idx2]
            file_names_1.append(fname1)
            file_names_2.append(fname2)
            idxs1.append(idx1)
            idxs2.append(idx2)

        dataloader = self.get_loftr_dataloader(file_names_1, file_names_2, idxs1, idxs2)

        with h5py.File(file_keypoints, mode='w') as f_match:
            store_mkpts = {}
            for X in tqdm(dataloader):
                image1, image2, key1, key2, idx1, idx2, ori_shape_1, ori_shape_2 = X
                fname1, fname2 = self.paths[idx1], self.paths[idx2]

            with torch.no_grad():
                correspondences = self.model({'image0': image1.to(self.device), 'image1': image2.to(self.device)})
                mkpts1 = correspondences['keypoints0'].cpu().numpy()
                mkpts2 = correspondences['keypoints1'].cpu().numpy()
                mconf = correspondences['confidence'].cpu().numpy()

            mkpts1[:, 0] *= (float(ori_shape_1[1]) / float(image1.shape[3]))
            mkpts1[:, 1] *= (float(ori_shape_1[0]) / float(image1.shape[2]))

            mkpts2[:, 0] *= (float(ori_shape_2[1]) / float(image2.shape[3]))
            mkpts2[:, 1] *= (float(ori_shape_2[0]) / float(image2.shape[2]))

            n_matches = mconf.shape[0]

            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2, data=np.concatenate([mkpts1, mkpts2], axis=1).astype(np.float32))


class Merge:
    def __init__(self, paths: list[Path], index_pairs: list[tuple[int, int]], files_keypoints: list, \
                 feature_dir: Path, device: torch.device = torch.device('cpu')):
        self.paths = paths
        self.index_pairs = index_pairs
        self.files_keypoints = files_keypoints
        self.feature_dir = feature_dir
        self.device = device

    def get_unique_idxs(self, A, dim=0):
        unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
        first_indices = ind_sorted[cum_sum]
        return first_indices

    def get_keypoint_from_h5(self, fp, key1, key2):
        rc = -1
        try:
            kpts = np.array(fp[key1][key2])
            rc = 0
            return (rc, kpts)
        except:
            return (rc, None)

    def get_keypoint_from_multi_h5(self, fps, key1, key2):
        list_mkpts = []
        for fp in fps:
            rc, mkpts = self.get_keypoint_from_h5(fp, key1, key2)
            if rc == 0:
                list_mkpts.append(mkpts)
        if len(list_mkpts) > 0:
            list_mkpts = np.concatenate(list_mkpts, axis=0)
        else:
            list_mkpts = None
        return list_mkpts

    def keypoints_merger(self) -> None:
        fps = [h5py.File(file, mode='r') for file in self.files_keypoints]

        with h5py.File(f'{self.feature_dir}/merge_tmp.h5', mode='w') as f_match:

            for pair_idx in progress_bar(self.index_pairs):
                idx1, idx2 = pair_idx
                fname1, fname2 = self.paths[idx1], self.paths[idx2]
                key1, key2 = fname1.name, fname2.name

                mkpts = self.get_keypoint_from_multi_h5(fps, key1, key2)
                if mkpts is None:
                    continue
                group = f_match.require_group(key1)
                group.create_dataset(key2, data=mkpts)

        for fp in fps:
            fp.close()

        kpts = defaultdict(list)
        match_indexes = defaultdict(dict)
        total_kpts = defaultdict(int)
        with h5py.File(f'{self.feature_dir}/merge_tmp.h5', mode='r') as f_match:
            for k1 in f_match.keys():
                group = f_match[k1]
                for k2 in group.keys():
                    matches = group[k2][...]
                    total_kpts[k1]
                    kpts[k1].append(matches[:, :2])
                    kpts[k2].append(matches[:, 2:])
                    current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                    current_match[:, 0] += total_kpts[k1]
                    current_match[:, 1] += total_kpts[k2]
                    total_kpts[k1] += len(matches)
                    total_kpts[k2] += len(matches)
                    match_indexes[k1][k2] = current_match

        for k in kpts.keys():
            kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
        unique_kpts = {}
        unique_match_idxs = {}
        out_match = defaultdict(dict)
        for k in kpts.keys():
            uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]), dim=0, return_inverse=True)
            unique_match_idxs[k] = uniq_reverse_idxs
            unique_kpts[k] = uniq_kps.numpy()
        for k1, group in match_indexes.items():
            for k2, m in group.items():
                m2 = deepcopy(m)
                m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
                m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
                mkpts = np.concatenate([unique_kpts[k1][m2[:, 0]], unique_kpts[k2][m2[:, 1]], ], axis=1)
                unique_idxs_current = self.get_unique_idxs(torch.from_numpy(mkpts), dim=0)
                m2_semiclean = m2[unique_idxs_current]
                unique_idxs_current1 = self.get_unique_idxs(m2_semiclean[:, 0], dim=0)
                m2_semiclean = m2_semiclean[unique_idxs_current1]
                unique_idxs_current2 = self.get_unique_idxs(m2_semiclean[:, 1], dim=0)
                m2_semiclean2 = m2_semiclean[unique_idxs_current2]
                out_match[k1][k2] = m2_semiclean2.numpy()

        with h5py.File(f'{self.feature_dir}/keypoints.h5', mode='w') as f_kp:
            for k, kpts1 in unique_kpts.items():
                f_kp[k] = kpts1

        with h5py.File(f'{self.feature_dir}/matches.h5', mode='w') as f_match:
            for k1, gr in out_match.items():
                group = f_match.require_group(k1)
                for k2, match in gr.items():
                    group[k2] = match
