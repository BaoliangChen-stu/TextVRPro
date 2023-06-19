#Cross-Modal Video Retrieval Assisted with Image Text Matching

**Cross-Modal Video Retrieval Assisted with Image Text Matching is divided into two steps.Firstly,We train the baseline model
TextVR at video level.Then,we use CLIP to match the captions and video during test stage.**

##Step1: Train TextVR Model
1.1 Prepare the environment, data, and model according to official instructions for TextVR

`python train.py -c configs/TextVR_fusion.json
`

1.2 Inference TextVR 

python `test.py -c configs/TextVR_fusion.json -r ckpt/textvr.pth --sim_path sim_matrix.npy`

##Step2: Convert test data into images
2.1 The code for convert Video to Image in `videoUtils.py`

##Step3: Matching Video and captions in test Videos with CLIP
The code use CLIP for Image of Video in `./Image/video_captions_match.py` Then we need to convert the result from video2text to
text2video use convertRes() in `video_captions_match.py`

##Step4: Result Fusion
We use an average weighting strategy for the two results obtained by TextVR and CLIP.
