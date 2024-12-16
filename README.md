# PSAR-SR
The success of the ClassSR has led to a strategy of decomposing images being used for large image SR. The decomposed image patches have different recovery difficulties. Therefore, in ClassSR, image patches are reconstructed by different networks to greatly reduce the computational cost. However, in ClassSR, the training of multiple sub-networks inevitably increases the training difficulty. Furthermore, decomposing images with overlapping not only increases the computational cost but also inevitably produces artifacts. To address these challenges, we propose an end-to-end general framework, named patches separation and artifacts removal SR (PSAR-SR). In PSAR-SR, we propose an image information complexity module (IICM) to efficiently determine the difficulty of recovering image patches. Then, we propose a patches classification and separation module (PCSM), which can dynamically select an appropriate SR path for image patches of different recovery difficulties. Moreover, we propose a multi-attention artifacts removal module (MARM) in the network backend, which can not only greatly reduce the computational cost but also solve the artifacts problem well under the overlapping-free decomposition. Further, we propose two loss functions - threshold penalty loss (TP-Loss) and artifacts removal loss (AR-Loss). TP-Loss can better select appropriate SR paths for image patches. AR-Loss can effectively guarantee the reconstruction quality between image patches. Experiments show that compared to the leading methods, PSAR-SR well eliminates artifacts under the overlapping-free decomposition and achieves superior performance on existing methods (e.g., FSRCNN, CARN, SRResNet, RCAN and CAMixerSR). Moreover, PSAR-SR saves 53% - 65% FLOPs in computational cost far beyond the leading methods.

Dependencies
--
* python >= 3.6.0
* pytorch >= 1.7.0
* torchvision >= 0.8.0

Dataset
--
Download Training Set ([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)) and Testing Set ([Test2K, 4K, 8K](https://drive.google.com/drive/folders/18b3QKaDJdrd9y0KwtrWU2Vp9nHxvfTZH))

Training
--   
1. Densely crop the training set according to the paper description
2. Revise the ``dataset_dir``, ``eval_file`` and ``evallabel_file`` in ``main_torch.py``  
3. Run ``main_torch.py``  

Testing
--
1. Revise the ``weights_file``, ``test_file``, ``testlabel_file`` and ``save_file`` in ``test_torch.py``  
2. Run ``test_torch.py``
