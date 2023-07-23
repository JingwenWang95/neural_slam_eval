# Neural SLAM Evaluation Benchmark
This repo contains evaluation code for Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM, a neural SLAM method that perform real-time camera tracking and dense reconstruction based on a joint encoding.

### [Project Page](https://hengyiwang.github.io/projects/CoSLAM) | [Paper](https://arxiv.org/abs/2304.14377)
> Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM <br />
> [Hengyi Wang*](https://hengyiwang.github.io/), [Jingwen Wang*](https://jingwenwang95.github.io/), [Lourdes Agapito](http://www0.cs.ucl.ac.uk/staff/L.Agapito/) <br />
> CVPR 2023

<p align="center">
  <a href="">
    <img src="./media/coslam_teaser.gif" alt="Logo" width="80%">
  </a>
</p>

In this repo we also provide a comprehensive comparison for existing open-sourced RGB-D Neural SLAM methods under the same evaluation protocol. We hope this will benefit the research in the area of Neural SLAM.

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#datasets">Datasets</a>
    </li>
    <li>
      <a href="#evaluation-protocol">Evaluation Protocol</a>
    </li>
    <li>
      <a href="#run-evaluation">Run Evaluation</a>
    </li>
    <li>
      <a href="#benchmark">Benchmark</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## Installation
This repo assumes you already configured the environment from [Co-SLAM](https://github.com/HengyiWang/Co-SLAM) main repository. You then also need the following dependencies:
* Open3D
* pyglet
* pyrender

You can install those dependencies by running:

```bash
conda activate coslam
pip install -r requirements.txt
```

## Datasets
Following iMAP and NICE-SLAM we evaluate our method on Replica, ScanNet and TUM RGB-D datasets. 
We perform further experiments on the synthetic dataset from NeuralRGBD which contains many thin structures and simulates the noise present in real depth sensor measurements.
We provided the download links in our [main repo](https://github.com/HengyiWang/Co-SLAM).

In addition to those sequences and the apartment sequences captured by the authors of [NICE-SLAM](https://github.com/cvg/nice-slam/blob/master/scripts/download_apartment.sh), we also collected our own real-world indoor scenes (MyRoom) using RealSense D435i depth camera, 
which is more popular among robotics community and whose depth quality is slightly worse than Azure Kinect. You can download MyRoom sequences (~15G) from [here](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabjw4_ucl_ac_uk/EW-_dVBx8pFImN-nzlL7d9YB9ikr_GMkI339cSFK4lsFWw?e=d1ksWJ).

## Evaluation Protocol

### Mesh Culling
As stated in Section 1.2 of our [supplementary material](https://arxiv.org/abs/2304.14377), for neural SLAM methods mesh culling is required for evaluating the reconstruction quality due to the extrapolation ability of neural networks.
This extrapolation property brings the hole-filling ability to neural SLAM methods but also could potentially produce unwanted artefacts outside the region of interest (ROI). 
Ideally we want a culling strategy that could remove unwanted part outside the ROI and leave all other parts unchanged.

Existing culling methods used in neural SLAM/reconstruction systems are either based on ***frustum*** (NICE-SLAM and iMAP) or ***frustum+occlusion*** (Neural-RGBD and GO-Surf) strategy.
The first might leave artefacts outside the ROI (such as artefacts behind the wall) while the second remove the occluded parts inside the ROI. 
In Co-SLAM, we propose to further use ***frustum+oclusion+virtual cameras*** that introduces extra virtual views that cover the occluded parts inside the region of interest. 
Please refer to Section 1.2 of our [supplementary material](https://arxiv.org/abs/2304.14377) for more explanation and details.

We provide our culling script that subsumes all three culling strategies mentioned above in case you want to follow the other two protocols. Here is an example usage:
```bash
INPUT_MESH=output/Replica/office0/mesh_final.ply
python cull_mesh.py --config configs/Replica/office0.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --gt_pose  # Co-SLAM strategy
python cull_mesh.py --config configs/Replica/office0.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose  # Neural-RGBD/GO-Surf strategy
python cull_mesh.py --config configs/Replica/office0.yaml --input_mesh $INPUT_MESH --gt_pose  # iMAP/NICE-SLAM strategy
```

### Command line arguments
- `--config $CONFIG_FILE` config file for the scene of input mesh
- `--input_mesh $INPUT_MESH` path to the mesh to be culled
- `--output_mesh $OUTPUT_MESH` path to save the culled mesh (optional)
- `--remove_occlusion` remove self-occlusion or not
- `--virt_cam_path` path to virtual cameras directory (optional)
- `--virtual_cameras` use virtual cameras or not
- `--gt_pose` use GT poses or not
- `--ckpt_path` path to reconstructed checkpoint (optional)

Note that to use Co-SLAM culling strategy you need virtual camera views. 
We provide data required to evaluate Co-SLAM on both Replica and Neural-RGBD dataset.


### Virtual Camera Views
The purpose of virtual views is just to cover regions that might not be observed by existing views, so their selection is very flexible. 
To give you a flavour of how it can be done, here we also provide a simple example Python script to create virtual cameras for Replica sequences in an interactive fashion:
```bash
python create_virtual_cameras_replica.py --config configs/Replica/office0.yaml --data_dir data/Replica/office0
```
It will first create TSDF-Fusion mesh with GT poses. Once the mesh is created an Open3D window will pop up, you can adjust the view-point using your mouse to cover the unobserved regions. Press `.` button on the keyboard to save the view-point.

## Run Evaluation

### Reconstruction
To evaluate reconstruction quality, first download the data needed for evaluation:
* [Replica](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabjw4_ucl_ac_uk/EUfNXQ_qps5DtYJP7FNegxQBHoQPUpg63TcVOUzFAubeDQ?e=aGTFrp)
* [SyntheticRGBD](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabjw4_ucl_ac_uk/EW_AaEdHND1ElCLK0GXLBXgBND4iKy_mKza0Xf8GxQYq5w?e=9N0kTI)

which contains virtual camera views, GT camera meshes culled with those virtual views using our proposed culling strategy (for 3D-metric) and unseen points on the GT meshes (for 2D-metric). 
For reproducibility purpose we also included the sampled 1000 camera poses for 2D evaluation. 

```
<scene_name>           
├── virtual_cameras             # virtual cameras
    ├── 0.txt     
    ├── 0.png
    ├── 1.txt
    ├── 1.png
    ...
├── sampled_poses_1000.npz      # sampled 1000 camrera poses
├── gt_pc_unseen.npy            # point cloud of unseen part
├── gt_unseen.ply               # mesh of unseen part
├── gt_mesh_cull_virt_cams.ply  # culled ground-truth mesh
```

Then run the culling script to cull the reconstructed mesh
```bash
# Put your own path to reconstructed mesh. Here is just an example
INPUT_MESH=output/Replica/office0/mesh_final.ply
VIRT_CAM_PATH=eval_data/Replica/office0/virtual_cameras
python cull_mesh.py --config configs/Replica/office0.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose  # Co-SLAM strategy
```
Once you've got the culled reconstructed mesh, the evaluation follows similar pipeline as iMAP/NICE-SLAM. 
```bash
REC_MESH=output/Replica/office0/mesh_final_cull_virt_cams.ply
GT_MESH=eval_data/Replica/office0/gt_mesh_cull_virt_cams.ply)
python eval_recon.py python --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type Replica -2d -3d
```

### Tracking
We follow exactly the same evaluation protocol for evaluating average trajectory error (ATE). 
Please refer to [NICE-SLAM](https://github.com/cvg/nice-slam/tree/master) or our [main repo](https://github.com/HengyiWang/Co-SLAM) for more details.

## Benchmark
In this section we compare other methods on reconstruction quality, tracking accuracy and performance analysis.
All performance analysis were conducted on the same computing platform: a desktop PC with a 3.60GHz Intel Core i7-12700K CPU and a single NVIDIA RTX 3090ti GPU.
To rule out the effect of method-dependent implementation details such as data loading, different multi-processing strategy, we only report the time needed to perform tracking/mapping iterations and the corresponding FPS.
We also report total time needed to process each sequence in individual dataset page under each section.

### Replica

|     Methods      | Acc↓<br/>[cm] | Comp↓<br/>[cm] | Comp<br/>Ratio↑<br/>[%] | Depth <br/>L1↓<br/>[cm] | Track.↓<br/>[ms x it] | Map.↓<br/>[ms x it] | Track.<br/>FPS↑ | Map.<br/>FPS↑ | #param↓ | 
|:----------------:|:-------------:|:--------------:|:-----------------------:|:-----------------------:|:---------------------:|:-------------------:|:---------------:|:-------------:|:-------:|
|       iMAP       |     3.62      |      4.93      |          80.51          |          4.64           |        16.8x6         |       44.8x10       |      9.92       |     2.23      |  0.26M  |
|    NICE-SLAM     |     2.37      |      2.64      |          91.13          |          1.90           |        7.8x10         |       82.5x60       |      13.70      |     0.20      |  17.4M  |
|    Vox-Fusion    |     1.88      |      2.56      |          90.93          |          2.91           |        15.8x30        |       46.0x10       |      2.11       |     2.17      |  0.87M  |
|      ESLAM       |     2.18      |      1.75      |          96.46          |          0.94           |         6.9x8         |       18.4x15       |      18.11      |     3.62      |  9.29M  |
|     Co-SLAM      |     2.10      |      2.08      |          93.44          |          1.51           |        5.8x10         |       9.8x10        |      17.24      |     10.20     |  0.26M  |

Here tracking/mapping FPS indicates how fast a complete tracking/mapping optimization cycle can run, thus do not correspond to the actual runtime FPS of the system. 
For the overall system runtime we report the total time needed to process an entire sequence in [benchmark/replica](benchmark/replica). 
Also note that on Replica mapping happens ~every 5 frames for iMAP*, NICE-SLAM and Co-SLAM. 
Vox-Fusion adopts different multi-processing strategy and mapping is performed as frequently as possible.

Please refer to [benchmark/replica](benchmark/replica) for more details and breakdown of each scene.

### SyntheticRGBD

|  Methods   | Acc↓<br/>[cm] | Comp↓<br/>[cm] | Comp<br/>Ratio↑<br/>[%] | Depth <br/>L1↓<br/>[cm] | Track.↓<br/>[ms x it] | Map.↓<br/>[ms x it] | Track.<br/>FPS↑ | Map.<br/>FPS↑ | #param↓ | 
|:----------:|:-------------:|:--------------:|:-----------------------:|:-----------------------:|:---------------------:|:-------------------:|:---------------:|:-------------:|:-------:|
|   iMAP*    |     18.29     |     26.41      |          20.73          |          47.22          |        31.0x50        |      49.1x300       |      0.64       |     0.07      |  0.22M  |
| NICE-SLAM  |     5.95      |      5.30      |          77.46          |          6.32           |        12.3x10        |       50.4x60       |      8.13       |     0.33      |  3.11M  |
| Vox-Fusion |     4.10      |      4.81      |          81.78          |          6.13           |        16.6x30        |       46.2x10       |      2.00       |     2.16      |  0.84M  |
|  Co-SLAM   |     2.95      |      2.96      |          86.88          |          3.02           |        6.4x10         |       10.4x10       |      15.63      |     9.62      |  0.26M  |

Here tracking/mapping FPS indicates how fast a complete tracking/mapping optimization cycle can run, thus do not correspond to the actual runtime FPS of the system.
For the overall system runtime we report the total time needed to process an entire sequence in [benchmark/rgbd](benchmark/rgbd).
All experiments are done with Replica seetings of each method.

Please refer to [benchmark/rgbd](benchmark/rgbd) for more details of each scene.

### ScanNet

|  Methods   | ATE↓<br/>[cm] | ATE↓<br/>w/o align<br/>[cm] | Track.↓<br/>[ms x it] | Map.↓<br/>[ms x it] | Track.<br/>FPS↑ | Map.<br/>FPS↑ | #param↓ |
|:----------:|:-------------:|:---------------------------:|:---------------------:|:-------------------:|:---------------:|:-------------:|:-------:|
|   iMAP*    |     36.67     |              -              |        30.4x50        |      44.9x300       |      0.66       |     0.07      |  0.2M   |
| NICE-SLAM  |     9.63      |            23.97            |        12.3x50        |      125.3x60       |      1.63       |     0.13      |  10.3M  |
| Vox-Fusion |     8.22      |              -              |        29.4x30        |       85.8x15       |      1.13       |     0.78      |  1.1M   |
|   ESLAM    |     7.42      |              -              |        7.4x30         |       22.4x30       |      4.54       |     1.49      |  10.5M  |
|  Co-SLAM   |     9.37      |            18.01            |        7.8x10         |       20.2x10       |      12.82      |     4.95      |  0.8M   |
|  Co-SLAM†  |     8.75      |              -              |        7.8x20         |       20.2x10       |      6.41       |     4.95      |  0.8M   |

Here tracking/mapping FPS indicates how fast a complete tracking/mapping optimization cycle can run, thus do not correspond to the actual runtime FPS of the system.
For the overall system runtime we report the total time needed to process an entire sequence in [benchmark/scannet](benchmark/scannet). 
Also note that on ScanNet mapping happens ~every 5 frames for iMAP*, NICE-SLAM and Co-SLAM.
Vox-Fusion adopts different multi-processing strategy and mapping is performed as frequently as possible.

Please refer to [benchmark/scannet](benchmark/scannet) for more details of each scene.

### TUM-RGBD

| Methods   | ATE↓<br/>[cm] | Track.↓<br/>[ms x it] | Map.↓<br/>[ms x it] | Track.<br/>FPS↑ | Map.<br/>FPS↑ | #param |
|-----------|:-------------:|:---------------------:|:-------------------:|:---------------:|:-------------:|:------:|
| iMAP      |     4.23      |           -           |          -          |        -        |       -       |   -    |
| iMAP*     |     6.10      |       29.6x200        |      44.3x300       |      0.17       |     0.08      |  0.2M  |
| NICE-SLAM |     2.50      |       47.1x200        |      189.2x60       |      0.11       |     0.09      | 101.6M |
| Co-SLAM   |     2.40      |        7.5x10         |       19.0x20       |      13.33      |     2.63      |  1.6M  |
| Co-SLAM†  |     2.17      |        7.5x20         |       19.0x20       |      6.67       |     2.63      |  1.6M  |

Here tracking/mapping FPS indicates how fast a complete tracking/mapping optimization cycle can run, thus do not correspond to the actual runtime FPS of the system.
For the overall system runtime we report the total time needed to process an entire sequence in [benchmark/tum](benchmark/tum).
Also note that on TUM-RGBD mapping happens ~every frame for NICE-SLAM and iMAP*, and ~every 5 frames for Co-SLAM.

Please refer to [benchmark/tum](benchmark/tum) for more details of each scene.

## Acknowledgement

This repository adapted codes from some awesome repositories, including [NICE-SLAM](https://github.com/cvg/nice-slam), [NeuralRGBD](https://github.com/dazinovic/neural-rgbd-surface-reconstruction) and [GO-Surf](https://github.com/JingwenWang95/go-surf). Thanks for making the code available. We also thank [Zihan Zhu](https://zzh2000.github.io/) of [NICE-SLAM](https://github.com/cvg/nice-slam), [Edgar Sucar](https://edgarsucar.github.io/) of [iMAP](https://edgarsucar.github.io/iMAP/) for quick response of details of their methods.

The research presented here has been supported by a sponsored research award from Cisco Research and the UCL Centre for Doctoral Training in Foundational AI under UKRI grant number EP/S021566/1. This project made use of time on Tier 2 HPC facility JADE2, funded by EPSRC (EP/T022205/1).

## Citation

If you find our code/work useful in your research or wish to refer to the benchmark results, please consider citing the following:

```bibtex
@article{wang2023co-slam,
  title={Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM},
  author={Wang, Hengyi and Wang, Jingwen and Agapito, Lourdes},
  journal={arXiv preprint arXiv:2304.14377},
  year={2023}
}

@inproceedings{wang2022go-surf,
  author={Wang, Jingwen and Bleja, Tymoteusz and Agapito, Lourdes},
  booktitle={2022 International Conference on 3D Vision (3DV)},
  title={GO-Surf: Neural Feature Grid Optimization for Fast, High-Fidelity RGB-D Surface
  Reconstruction},
  year={2022},
  pages = {433-442},
  organization={IEEE}
}
```

## Contact
Contact [Hengyi Wang](hengyi.wang.21@ucl.ac.uk) and [Jingwen Wang](mailto:jingwen.wang.17@ucl.ac.uk) for questions and reporting bugs.
