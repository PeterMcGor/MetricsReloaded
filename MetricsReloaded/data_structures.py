from abc import ABC, abstractmethod
from typing import Union, Tuple, List
import os
import inspect
#import multiprocessing
from multiprocessing.pool import Pool
from collections.abc import Iterable

import numpy as np
from pathlib import Path

from MetricsReloaded.processes.overall_process import ProcessEvaluation

class Category:
    IC = 'Image Classification'
    OD = 'Object Detection'
    SS = 'Semantic Segmentation'
    IS = 'Instance Segmentation'
    # dict to translate to the Metrics Reloaded references fo the categories
    MR_NOMENCLATURE = {IC:'ImLC', OD:'ObD', SS:'SemS', IS:'InS'}

class DataCase(ABC):
    """
    The base class to populate the needeed dictionary data per case. Otherwise is a mess to get use to the library
    """
    FILES = 'file'
    CASE_ID = 'case'
    LABELS = 'list_values'
    REF_BLOBS = 'ref_loc'
    PRE_BLOBS = 'pred_loc'
    REF_CLASSES = 'ref_class'
    PRE_CLASSES = 'pred_class'
    PRED_PROBA = 'pred_prob'
    
    
    def __init__(self, 
                 files:Union[Path, list[Path]], 
                 annotattion_labels: list[int], 
                 case_id:str = None,
                 reference_blobs: Union[list[np.ndarray], None]=None, 
                 predicted_blobs: Union[list[np.ndarray], None]=None, 
                 reference_class_per_blob: list[int]=None, 
                 predicted_class_per_blob: list[int]=None,  
                 predicted_proba_per_blob: list[float]=None):
        
        self.files = files
        self.case_id = case_id
        self.annotattion_labels = annotattion_labels
        self.reference_blobs = reference_blobs
        self.predicted_blobs = predicted_blobs
        self.reference_class_per_blob = reference_class_per_blob
        self.predicted_class_per_blob = predicted_class_per_blob
        self.predicted_proba_per_blob = predicted_proba_per_blob
        assert len(self.reference_blobs) == len(self.reference_class_per_blob), f"The number of blobs in the reference {len(self.reference_blobs)} is different to the number of the blob with a given class {len(self.reference_class_per_blob)}"
        assert len(self.predicted_blobs) == len(self.predicted_class_per_blob) == len(self.predicted_proba_per_blob), "The number of blobs in the predicted is different to the number of the blobs with a given class or probability"
    
    ### Getter and seetwr for case id ###
    @property
    def case_id(self) -> str:
        return self._case_id
    
    @case_id.setter
    def case_id(self, case_id:str = None):
        if case_id is None:
            self._case_id = self.files[0] if self.files is Iterable else self.files
        else:
            self._case_id = case_id
    
    ### Getter and seetwr for file ###
    @property
    def files(self) -> Union[Path, list[Path], Tuple[Path, Path],list[Tuple[Path,Path]] ]:
        return self._files
    
    @files.setter
    @abstractmethod
    def files(self, files:Union[Path, list[Path], list[Tuple[Path,Path]] ]):
        pass
    
    ### Getter and setter for the labels ###
    @property
    def annotattion_labels(self) -> list[int]:
        return self._annotattion_labels
    
    @annotattion_labels.setter
    @abstractmethod
    def annotattion_labels(self, annotattion_labels:list[int]):
        pass
    
    ### Getter and setter for the list of blobs in the refeence case ###
    @property
    def reference_blobs(self) -> list[np.ndarray]:
        return self._reference_blobs
    
    @reference_blobs.setter
    @abstractmethod
    def reference_blobs(self, reference_blobs:list[np.ndarray]):
        pass
    
    
    ### Getter and setter for the list of blobs in the predictions case ###
    @property
    def predicted_blobs(self) -> list[np.ndarray]:
        return self._predicted_blobs
    
    @predicted_blobs.setter
    @abstractmethod
    def predicted_blobs(self, predicted_blobs:list[np.ndarray]):
        pass
    
    
    
    ### Getter and setter for the list of reference classes in the case per blob ###
    @property
    def reference_class_per_blob(self) -> list[int]:
        return self._reference_class_per_blob
    
    @reference_class_per_blob.setter
    @abstractmethod
    def reference_class_per_blob(self, reference_class_per_blob:list[int]):
        pass
    
    
    ### Getter and setter for the list of predicted classes in the case per blob ###
    @property
    def predicted_class_per_blob(self) -> list[int]:
        return self._predicted_class_per_blob
    
    @predicted_class_per_blob.setter
    @abstractmethod
    def predicted_class_per_blob(self, predicted_class_per_blob:list[int]):
        pass
    
    
    ### Getter and setter for the list of probabilities in the case per blob ###
    @property
    def predicted_proba_per_blob(self) -> list[float]:
        return self._predicted_proba_per_blob
    
    @predicted_proba_per_blob.setter
    @abstractmethod
    def predicted_proba_per_blob(self, predicted_proba_per_blob:list[float]):
        pass
    
    
        
    def as_dict(self): 
        dct = {
            DataCase.FILES:self.files, 
            DataCase.LABELS: self.annotattion_labels, 
            DataCase.REF_BLOBS: self.reference_blobs, 
            DataCase.PRE_BLOBS: self.predicted_blobs, 
            DataCase.REF_CLASSES:self.reference_class_per_blob, 
            DataCase.PRE_CLASSES:self.predicted_class_per_blob, 
            DataCase.PRED_PROBA:self.predicted_proba_per_blob
        }
        return dct
    
    def as_PE_dict(self) -> dict:
        dct = self.as_dict()
        if isinstance(dct[DataCase.FILES], tuple): 
            dct[DataCase.FILES] = [dct[DataCase.FILES][0]]
            dct[DataCase.REF_BLOBS] = [dct[DataCase.REF_BLOBS]]
            dct[DataCase.REF_CLASSES] = [dct[DataCase.REF_CLASSES]]
            dct[DataCase.PRE_BLOBS] = [dct[DataCase.PRE_BLOBS]]
            dct[DataCase.PRE_CLASSES] = [dct[DataCase.PRE_CLASSES]]
            dct[DataCase.PRED_PROBA] = [dct[DataCase.PRED_PROBA]]
        return dct
    
    def run_process_evaluation(self, category:str, keep_blobs:bool=False, **kwargs):
        """
        """
        assert category in Category.MR_NOMENCLATURE.keys()
        evaluation = ProcessEvaluation(self.as_PE_dict(),
                                       Category.MR_NOMENCLATURE[category],
                                       file=self.as_PE_dict()[DataCase.FILES], case=False,**kwargs)
        self.detection_metrics = evaluation.resdet 
        self.segmentation_metrics = evaluation.resseg
        self.multithreshold_metrics =  evaluation.resmt 
        self.detailed_multithreshold_metrics = evaluation.resmt_complete
        self.multiclass_metrics = evaluation.resmcc
        self.calibration_metrics =  evaluation.rescal
        ## add the case_id ###
        for df in [self.detection_metrics, self.segmentation_metrics, self.multithreshold_metrics, self.detailed_multithreshold_metrics, self.multiclass_metrics, self.calibration_metrics]:
            if df is not None:
                df[DataCase.CASE_ID] = self.case_id 
                
        if not keep_blobs:
            self.reference_blobs = []
            self.predicted_blobs = []
        



from MetricsReloaded.utility.utils import MorphologyOps
import nibabel as nib

class DataCaseForSimpleIS(DataCase):
    """
    Create a data structure for a whole reference/prediction case in which the blobs are extracted
    employing MorpholicalOps and all the proba re the same. Predictin (and obviously refs.) can be given as labels
    """
    @DataCase.files.setter
    def files(self, files:Tuple[Path,Path]):
        # First Path is the reference image
        assert len(files) == 2, "Reference and prediction files are not paired"
        self.ref_img_path = files[0]
        self.pred_img_path =  files[1]
        self._files = files
        
    @DataCase.annotattion_labels.setter
    def annotattion_labels(self, labels):
        self._annotattion_labels = labels
        
    @DataCase.reference_blobs.setter
    def reference_blobs(self, blobs_list=None):
        if blobs_list is None:
            # the class storage all the independent blobs in the reference with a complementary list to point its label 
            reference_class_per_blob, blobs = get_blobs_per_label(nib.load(self.ref_img_path).get_fdata(), self.annotattion_labels)
            
            self._reference_blobs = blobs
            self.reference_class_per_blob = reference_class_per_blob
        else:
            self._reference_blobs = blobs_list
        
    @DataCase.predicted_blobs.setter
    def predicted_blobs(self, blobs_list):
        #Note that in this case the Class expects a file  in which the predictions are integer for the labels and not anything else
        if blobs_list is None:
            prediction = nib.load(self.pred_img_path).get_fdata()
            #print(np.unique(prediction))
            assert all(elem in np.unique(prediction) for elem in self.annotattion_labels), "Not all unique elements from prediction are in annotattion_labels."
            predicted_class_per_blob, blobs = get_blobs_per_label(prediction, self.annotattion_labels)
            self._predicted_blobs = blobs
            self.predicted_class_per_blob = predicted_class_per_blob
            self.predicted_proba_per_blob = [1] * len(blobs)
        else:
            self._predicted_blobs = blobs_list
        
    @DataCase.reference_class_per_blob.setter
    def reference_class_per_blob(self, class_list= None):
        if class_list is None:
            pass
        else:
            self._reference_class_per_blob = class_list
        
    @DataCase.predicted_class_per_blob.setter
    def predicted_class_per_blob(self, class_list):
        if class_list is None:
            pass
        else:
            self._predicted_class_per_blob = class_list
        
    @DataCase.predicted_proba_per_blob.setter
    def predicted_proba_per_blob(self, predicted_proba_per_blob_list):
        if predicted_proba_per_blob_list is None:
            pass
        else:
            self._predicted_proba_per_blob = predicted_proba_per_blob_list
            
            

class DataCaseAgg():
    def __init__(self, files:List[Tuple[Path, Path, str]], category:str=Category.IS, annotattion_labels:list = [1]):
        """
        file: a list compose by tuples with the ref. path, the prediction path and the case id.
        """
        self.files = files
        self.n_data_cases = len(self.files)
        self.data_cases = [None]*self.n_data_cases
        self.category = category
        self.annotattion_labels = annotattion_labels
        
    @property
    def files(self):
        return self._files
    @files.setter
    def files(self, files):
        assert isinstance(files, Iterable), "Files are not iterable"
        correct_files = []
        for file in files:
            assert len(file) >=2, f"The tuple {file} is not complete"
            assert all([os.path.exists(file[0]), os.path.exists(file[1])]), f"{file[0]} and/or {file[1]} are incorrect paths"
            if len(file)==2:
                correct_files.append((file[0], file[1], None))
            else:
                correct_files.append((file[0], file[1], file[2]))
        self._files = correct_files
        
    def populate_DataCase(self, indx:int, data_case:DataCase=DataCaseForSimpleIS, **kwargs):
        assert indx < len(self.data_cases)
        # Here the blobs are extracted so this operation is costly for the memory.
        self.data_cases[indx] = data_case(files=(self.files[indx][0], self.files[indx][1]), 
                                         case_id=self.files[indx][2],
                                         annotattion_labels=self.annotattion_labels, **kwargs)
        
    def run_process_evaluation(self, indx:int, **kwargs):
        assert indx < len(self.data_cases)
        assert self.data_cases[indx] is not None, f"The dataCase for {self.files[indx][0]} has not been populated previously. Maybe try populate_DataCase fun. for indx={indx}"
        self.data_cases[indx].run_process_evaluation(category=self.category, **kwargs)
        
    def populate_and_evaluate(self, indx:int, populate_args:dict={'data_case':DataCaseForSimpleIS}, evaluation_args:dict={"keep_blobs":False}):
        #print("Populate", populate_args,evaluation_args)
        self.populate_DataCase(indx, **populate_args)
        self.run_process_evaluation(indx, **evaluation_args)
        
    def populate_and_evaluate_all(self,kwargs):
        print("ALL", kwargs)
        for indx in range(self.n_data_cases):
            self.populate_and_evaluate(indx, **kwargs)
        
    
    def parallel_populate_and_evaluate(self, num_processes=4, **kwargs):
        # Create a multiprocessing pool with the desired number of processes
        #pool = multiprocessing.Pool(processes=num_processes)
        print("parallel", kwargs)
        with Pool(num_processes) as pool:
            # Map the populate_and_evaluate function to multiple indices in parallel
            data_cases_join = pool.starmap(self._parallel_populate_and_evaluate_helper, [(indx, kwargs) for indx in range(len(self.data_cases))])
        assert len(data_cases_join) == len(self.data_cases)
        for i, dc in enumerate(data_cases_join):
            self.data_cases[i] = dc


    def _parallel_populate_and_evaluate_helper(self, indx, kwargs):
        #data_case(files=(self.files[indx][0], self.files[indx][1]), 
        #                                 case_id=self.files[indx][2],
        #                                 annotattion_labels=self.annotattion_labels, **get_fun_kwargs(DataCase.__init__, **kwargs ))
        self.populate_and_evaluate(indx, **kwargs)
        return self.data_cases[indx]
            
        

def get_blobs_per_label(img:np.array, labels = [1]) -> (list[int], list[np.ndarray]):
    """
    Return two lists. Both of len(blobs). The first is the corresponding label, the second the blobs itself
    """
    blobs_dct = {}
    list_blobs_all = []
    list_correspondent_label_all = []
    for l in labels:
        #None cause connectivity does not care in this functions
        list_blobs, _, _ = MorphologyOps(img==l, None).list_foreground_component(include_center_of_mass=False, include_volumes=False)
        list_blobs_all += list_blobs
        list_correspondent_label_all += [l] * len(list_blobs)
    return list_correspondent_label_all,  list_blobs_all
        

def get_fun_kwargs(fun, **kwargs) -> dict:
    print("get fun args", kwargs)
    all_args = list(inspect.signature(fun).parameters)
    print("All args", fun,all_args, {k: kwargs[k] for k in kwargs.keys() if k in all_args})
    return {k: kwargs.pop(k) for k in dict(kwargs) if k in all_args}
    
        