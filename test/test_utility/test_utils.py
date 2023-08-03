import pytest
from MetricsReloaded.utility.utils import MorphologyOps
import os
from glob import glob
from monai.data import create_test_image_3d
import tempfile
import nibabel as nib
import numpy as np

from MetricsReloaded.processes.overall_process import ProcessEvaluation


def get_metrics_reloaded_dict(pth_ref, pth_pred):
    """Prepare input dictionary for MetricsReloaded package."""
    preds = []
    refs = []
    names = []
    for r, p in zip(pth_ref, pth_pred):
        print(r, p)
        name = r.split(os.sep)[-1].split(".nii.gz")[0]
        names.append(name)

        ref = nib.load(r).get_fdata()
        pred = nib.load(p).get_fdata()
        refs.append(ref)
        preds.append(pred)

    dict_file = {}
    dict_file["pred_loc"] = preds
    dict_file["ref_loc"] = refs
    #dict_file["pred_prob"] = preds
    #dict_file["ref_class"] = refs
    dict_file["pred_prob"] = [1] * len(preds)
    dict_file["ref_class"] =  np.random.randint(3, size=len(preds))
    dict_file["pred_class"] = preds
    dict_file["list_values"] = [1,2,3]
    dict_file["file"] = pth_pred
    dict_file["names"] = names

    return dict_file


def main(tempdir, img_size=96):
    #config.print_config()
    #logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Set patch size
    #patch_size = (int(img_size / 2.0),) * 3
    
    """

    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(5):
        im, seg = create_test_image_3d(img_size, img_size, img_size, num_seg_classes=3)

        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join('/data/Projects/MetricsReloaded/examples', f"im{i:d}.nii.gz"))

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join('/data/Projects/MetricsReloaded/examples', f"lab{i:d}.nii.gz"))
    """
    
    
    # Prepare MetricsReloaded input
    pth_ref = sorted(list(filter(os.path.isfile, glob(tempdir + os.sep + "lab*.nii.gz"))))
    pth_pred = sorted(list(filter(os.path.isfile, glob(tempdir + os.sep + "pred*.nii.gz"))))
    
    # Prepare input dictionary for MetricsReloaded package
    dict_file = get_metrics_reloaded_dict(pth_ref, pth_pred)
    print(len(dict_file["pred_loc"]), dict_file["pred_loc"][0].shape)
    # Run MetricsReloaded evaluation process
    PE = ProcessEvaluation(
        dict_file,
        "InS",
        localization="mask_iou",
        file=dict_file["file"],
        flag_map=True,
        assignment="greedy_matching",
        measures_overlap=[
            "numb_ref",
            "numb_pred",
            "numb_tp",
            "numb_fp",
            "numb_fn",
            "iou",
            "fbeta",
        ],
        measures_boundary=[
            "assd",
            "boundary_iou",
            "hd",
            "hd_perc",
            "masd",
            "nsd",
        ],
        case=True,
        thresh_ass=0.000001,
    )

    # Save results as CSV
    PE.resseg.to_csv(tempdir + os.sep + "results_metrics_reloaded_ins.csv")
    




if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = '/data/Projects/MetricsReloaded/examples'
        main(tempdir)