import logging
from collections import OrderedDict

import numpy as np

from detectron2.evaluation import SemSegEvaluator

from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)


class NovelSemSegEvaluator(SemSegEvaluator):
    """
    TODO Doku
    """

    def __init__(self, dataset_name, distributed=True, output_dir=None):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
        )
        self._metadata = MetadataCatalog.get(dataset_name)

        self._novel_ids: dict = getattr(self._metadata, "novel_ids", {})
        self._novel_class_names: list = self._metadata.stuff_classes  # nur die neuen
        self._novel_class_names: list = [self._novel_class_names[i] for i in self._novel_ids]

        print(f"[NovelSemSegEvaluator] dataset={dataset_name}")
        print(f"novel_ids={self._novel_ids}")
        print(f"novel_classes={self._novel_class_names}")

    #def process(self, inputs, outputs):
    #    for inp, out in zip(inputs, outputs):
    #        output = out["sem_seg"].argmax(dim=0).to(self._cpu_device)
    #        pred = np.array(output, dtype=np.int64)

    #        gt_path = self.input_file_to_gt_file[inp["file_name"]]
    #        with PathManager.open(gt_path, "rb") as f:
    #            gt = np.array(Image.open(f), dtype=np.int64)

    #        remapped_gt = gt.copy()
    #        remapped_gt[gt == self._ignore_label] = self._num_classes

    #        self._conf_matrix += np.bincount(
    #            (self._num_classes + 1) * pred.reshape(-1) + remapped_gt.reshape(-1),
    #            minlength=self._conf_matrix.size,
    #            ).reshape(self._conf_matrix.shape)

    def evaluate(self):
        # parent macht bereits viele berechnungen
        parent_results = super().evaluate()

        if parent_results is None:
            return None

        res = parent_results["sem_seg"]  # hat bereits mIoU, mACC und so für Alle Klassen (gemittelt)

        # Umbenennen
        res["mIoU-All-Classes"] = res.pop("mIoU")
        res["mACC-All-Classes"] = res.pop("mACC")

        # unnötige einträge aussortieren
        keys_to_delete = [k for k in res if any(
            k.startswith(prefix) for prefix in ["IoU-", "ACC-", "BoundaryIoU-", "min(IoU"]
        )]
        for k in keys_to_delete:
            del res[k]

        # für die neuen klassen, gemittelte mIoU Werte berechnen
        tp       = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt   = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)

        iou_per_class = np.full(self._num_classes, np.nan)
        acc_per_class = np.full(self._num_classes, np.nan)

        for i in self._novel_ids:  # TODO noch mal gegenprüfen
            union = pos_gt[i] + pos_pred[i] - tp[i]
            if union > 0:
                iou_per_class[i] = tp[i] / union
            if pos_gt[i] > 0:
                acc_per_class[i] = tp[i] / pos_gt[i]

        valid_iou = [iou_per_class[i] for i in self._novel_ids if not np.isnan(iou_per_class[i])]
        valid_acc = [acc_per_class[i] for i in self._novel_ids if not np.isnan(acc_per_class[i])]

        res["mIoU-Novel-Classes"] = 100 * np.mean(valid_iou) if valid_iou else 0.0
        res["mACC-Novel-Classes"] = 100 * np.mean(valid_acc) if valid_acc else 0.0

        for i, name in zip(self._novel_ids, self._novel_class_names):
            res[f"IoU-{name}"] = 100 * iou_per_class[i] if not np.isnan(iou_per_class[i]) else 0.0
            res[f"ACC-{name}"] = 100 * acc_per_class[i] if not np.isnan(acc_per_class[i]) else 0.0

        logger.info(f"[NovelSemSegEvaluator] Results: {res}")
        return OrderedDict({"sem_seg": res})