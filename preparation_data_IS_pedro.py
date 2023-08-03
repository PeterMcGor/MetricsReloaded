import glob
from MetricsReloaded.processes.overall_process import ProcessEvaluation
import os
import nibabel as nib
import pickle as pkl
import pandas as pd
from MetricsReloaded.utility.utils import MorphologyOps
from MetricsReloaded.metrics.prob_pairwise_measures import ProbabilityPairwiseMeasures

from MetricsReloaded.utility.assignment_localization import AssignmentMapping

import numpy as np
np.random.seed(42)

list_reffile = glob.glob("examples/Ref/*")
list_predfile = glob.glob("examples/Pred/*")
list_det = []
list_seg = []
list_mt = []
print("List pred file", list_predfile)
for f in list_reffile:
    print('Ref. file', f)
    name = os.path.split(f)[1]
    name = name.split("Ref")[0]
    #name = name.split("Lesion_")[1]
    if not os.path.exists("examples/results/Det94AvFin_%s.csv" % name):

        list_pospred = [c for c in list_predfile if name in c]
        print("List PostPred", list_pospred)
       
        if len(list_pospred) == 1:
            ref = nib.load(f).get_fdata()
            print("LIST postpred == 1",list_pospred[0])
            pred = nib.load(list_pospred[0]).get_fdata()
            # pred = nib.load('examples/Pred/P112_TP2.nii.gz').get_fdata()
            # ref = nib.load('/Users/csudre/Data/B-RAPIDD/CLA66/CorrectLesion_B-RAP_0007_01_CLA66.nii.gz').get_fdata()
            # pred = nib.load('/Users/csudre/Data/B-RAPIDD/RAP66/CorrectLesion_B-RAP_0007_01_RAP66.nii.gz').get_fdata()
            ref_bin = ref >= 0.5
            pred_bin = pred >= 0.5
  
            list_ref, _, _ = MorphologyOps(ref_bin, 6).list_foreground_component()
            list_pred, _, _ = MorphologyOps(pred_bin, 6).list_foreground_component()
        
        
            print("LIST POSTPRED", list_pospred)
            print('Blobs in REF:', len(list_ref), 'Blobs in PRE:', len(list_pred))
            pred_prob = []
            pred_class = []
            ref_class = []
            for k in list_pred:
                pred_prob.append(1 * np.random.rand())
                #pred_prob.append(1)
                pred_class.append(int(pred_prob[-1] >= 0.0))
            ##PEDRO
            pred_class = np.asarray(pred_class)
            pred_prob = np.asarray(pred_prob)
       
            #ppm = ProbabilityPairwiseMeasures(pred_class, pred_prob)
            #print("-----PPM-------")
            #print(ppm.sensitivity_at_ppv(), ppm.froc())
            ##################
            for k in list_ref:
                ref_class.append(1)
             

            list_values = [1]
            file = list_pospred
            dict_file = {}
            dict_file["pred_loc"] = [list_pred]
            dict_file["ref_loc"] = [list_ref]
            dict_file["pred_prob"] = [pred_prob]
            dict_file["ref_class"] = [ref_class]
            dict_file["pred_class"] = [pred_class]
            dict_file["list_values"] = list_values
            dict_file["file"] = file
          
            # f = open("TestDataBRAP_%s.pkl"%name, "wb")  # Pickle file is newly created where foo1.py is
            # pkl.dump(dict_file, f)  # dump data to f
            # f.close()
        
            #AM = AssignmentMapping(pred_loc=list_pred, ref_loc=list_ref, pred_prob=pred_prob)
         
            PE = ProcessEvaluation(
                dict_file,
                "InS",
                localization="mask_iou",
                file=list_pospred,
                flag_map=True,
                assignment="greedy_matching",
                case=False,
                measures_overlap=[
                    "fbeta",
                    "numb_ref",
                    "numb_pred",
                    "numb_tp",
                    "numb_fp",
                    "numb_fn",
                    'dsc'
                ],
                measures_mcc=[],
                measures_pcc=[
                    "fbeta",
                    "numb_ref",
                    "numb_pred",
                    "numb_tp",
                    "numb_fp",
                    "numb_fn",
                ],
                measures_mt=['froc', 
                             'sens@ppv',
                             'ppv@sens',
                             'fppi@sens',
                             'sens@fppi'],
                measures_boundary=["masd", "nsd", "boundary_iou"],
                measures_detseg=["pq"],
                thresh_ass=0.000001, 
                #dict_args = {"value_sensitivity":0.5,"value_fppi":0.7}
            )
            print("PE", PE.resseg)
            #df_resdet, df_resseg, df_resmt, df_resmcc = PE.process_data()
            PE.resdet["id"] = name
            PE.resseg["id"] = name
            PE.resmt["id"] = name
            PE.resmt_complete['id'] = name
            PE.resdet.to_csv("examples/results/Det94AvFin_%s.csv" % name)
            PE.resseg.to_csv("examples/results/Seg94AvFin_%s.csv" % name)
            PE.resmt.to_csv("examples/results/MT94AvFin_%s.csv" % name)
            PE.resmt_complete.to_csv("examples/results/MTCom94AvFin_%s.csv" % name)
            #PE.resmcc.to_csv("examples/results/MCC94AvFin_%s.csv" % name)
            #if PE.case:
             #   PE.stats_lab.to_csv("examples/results/Stats_Lab-AvFin_%s.csv" % name)
             #   PE.stats_all.to_csv("examples/results/Stats_ALL-AvFin_%s.csv" % name)            

            list_det.append(PE.resdet)
            list_seg.append(PE.resseg)
            list_mt.append(PE.resmt)
df_resdetall = pd.concat(list_det)
df_ressegall = pd.concat(list_seg)
df_resmtall = pd.concat(list_mt)
df_resdetall.to_csv("examples/results/Det94AvFin.csv")
df_ressegall.to_csv("examples/results/Seg94AvFin.csv")
df_resmtall.to_csv("examples/results/MT94AvFin.csv") 
print(PE.resdet, PE.resseg)
