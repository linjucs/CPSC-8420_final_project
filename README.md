# CPSC-8420_final_project
Course project for CPSC-8420

# Training

## NMF based speech enhancement
`python main_NMF.py`
## DNN based speech enhancement
`python main_spec.py`
## auto-encoder based speech enhancement
`python main_ae.py`

# Testing
## DNN testing
`python clean.py --test_dir data/test/mix --enh_dir data/test/enhanced`
## AE testing
`python clean_time.py --test_dir data/test/mix --enh_dir data/test/enhanced`

# Acknowledgement
This code is partially based on the following project:
* [SEDNN](https://github.com/yongxuUSTC/sednn)
* [DNN_NMF](https://github.com/eesungkim/Speech_Enhancement_DNN_NMF)
