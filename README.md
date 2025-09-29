# CGDS
The above files constitute the CGDS gait recognition system. 
The CGDS_mat.m file is the data preprocessing file; 
The RES_CAT_model.py and RES_KongDong.py files represent the model architectures of CGDS_Net and CGDS_LightNet, respectively, with RES_KongDong.py simple adjustments, other architectural combinations can also be experimented with; 
The sV_10.py file contains the model training code.
# DataSet
The original data collected by TI's IWR2243 is a bin file, which can be read from the above readDCA1000_1.m, and the set parameters are CGDS_mat.m visible. As the original data collected by us is used for further research in the future, the radar data collection uses single transmit and multiple receive mode, and the data volume is large. Therefore, the data is put in the web disk. The link is:https://pan.baidu.com/s/1weonr_zaDAEHimfaFXAi2Q?pwd=1122 Extraction code: 1122

The preprocessed dataset of 10 individuals has been uploaded to: https://doi.org/10.5281/zenodo.17218184, which is compressed and can be restored from the aforementioned yasuo-huifu.py. The file's data precision has been changed from float64 to float32. After restoration, the file contains heatmaps (single-step), heatmaps5 (multi-step), and add/ file name (hyperparameters). The uncompressed file has been uploaded to:
