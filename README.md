# Video Reality Test

- Paper: [paper]()
- Project Page: [homepage](https://zhanyi1.github.io/VideoRealityTest/) Thanks zz for your contribution!😊😊😊, nice work with you!😘😚😘
- Dataset: [hugginface](https://huggingface.co/datasets/kolerk/Video_Reality_Test), [modelscope](https://modelscope.cn/datasets/wjqkoko/Video_Reality_Test)

## 1. Brief Introduction
Recent video generators can already mimic real footage, forcing us to rethink how we detect AI-produced clips—especially when audio is involved. To this end, we propose the benchmark Video Reality Test , which tackles this gap with ASMR-origin videos where subtle action–object cues matter and sound must stay locked to the visuals. Models play the “creator” role while VLMs act as reviewers, mirroring a peer-review loop that stresses perceptual realism rather than coarse classification. Experiments reveal how far systems like Veo3.1-Fast can go (fooling many VLMs) and how even strong reviewers such as Gemini 2.5-Pro still trail human experts, highlighting remaining weaknesses around audio–visual alignment and watermark reliance.

## 2. Todo List
- [x] Public paper
- [x] Public dataset
- [x] Public video understanding evaluation code
- [ ] Publish video generation code

## 3. Dataset Introduction
We release the complete ASMR corpus: real videos, extracted images, prompts, and outputs from 13 different video-generation settings (OpenSoraV2, Wan2.2, Sora2 variants, Veo3.1-fast, Diffsynth-Studio Hunyuan/StepFun, etc.). For each of the 149 scenes we therefore provide `1 + k` clips (with `k = 13` fakery families), enabling fine-grained studies of how creators vary while sharing identical textual grounding. Both ModelScope and Hugging Face mirrors host identical content; pick whichever CDN suits your location.

The layout below shows how the data is organized once `Video_Reality_Test.tar.gz` (or the ModelScope folder) is unpacked.

### Layout
- `Video_Reality_Test.tar.gz` — monolithic archive containing every real video, generated video, and metadata file. Use `tar -xzf Video_Reality_Test.tar.gz` to recreate the folder layout described below.
- Folder layout (already unpacked on the ModelScope repo) mirrors the archive so you can rsync individual generators without downloading the full tarball.

```
Video_Reality_Test/
├── Video_Reality_Test.tar.gz   # full archive; contains every file below
├── jq_1/                       # unpacked dataset root (remove stray __MACOSX)
│   ├── HunyuanVideo/           # Diffsynth-Studio → Hunyuan generations
│   ├── OpensoraV2/             # OpenSora V2 baselines
│   ├── Real_ASMR/              # real ASMR reference videos (+optional keyframes)
│   ├── Real_ASMR_Prompt.csv    # prompt sheet; ref=video filename, text=description
│   ├── Sora2-it2v/             # Sora2 image-to-video outputs
│   ├── Sora2-it2v-wo-watermark/# watermark-free variant of the above
│   ├── Sora2-t2v/              # Sora2 text-to-video runs
│   ├── StepVideo-t2v/          # Diffsynth-Studio → StepFun generations
│   ├── Veo3.1-fast/            # Veo 3.1 fast generations
│   ├── Wan2.2/                 # Wan 2.2 outputs
└── ...
```

Every generator-specific directory contains clips named after their prompt IDs so you can align them with `Real_ASMR_Prompt.csv`. The `__MACOSX` folder is safe to delete—it is added automatically by macOS archivers and contains no useful data.

## 4. Generation Setup
- `OpenSoraV2` (https://github.com/hpcaitech/Open-Sora) provided most baseline trajectories.
- `Wan2.2` (https://github.com/Wan-Video/Wan2.2) complemented cinematic scenes needing richer lighting.
- `Diffsynth-Studio` generated both `Hunyuan` and `StepFun` variants from identical prompts to compare vendor-specific biases.
- `Sora 2` clips were authored via the official portal at https://openai.com/sora.
- `Veo 3.1 fast` generations came from Google's preview interface at https://deepmind.google/technologies/veo/.

Unless otherwise noted, we kept the native sampler settings of each platform so downstream evaluators see the exact outputs human raters inspected.

## 5. Run the Evaluation Code
1. Install dependencies:
   ```bash
   conda create -n vrt python=3.10 -y
   conda activate vrt
   pip install -r requirements.txt
   ```
2. Download a dataset split (choose one link at the top) and extract it under `data/`. Update the `data_path` in `eval_judgement.py` and `eval_judgement_audio.py` so the scripts can locate the unpacked files.
3. Open `eval_judgement.py` and `eval_judgement_audio.py`, set the required API key/token variables and `MODEL_NAME` placeholders at the top of each file to match the provider you are evaluating. Without this step the scripts will exit immediately:
   ```python
   api_key = "your_api_key_here"
   model_name = "gemini-2.5-flash"
   ```
4. Launch the evaluators:
   ```bash
   # video reality test for visual only
   python eval_judgement.py 

   # video reality test for visual+audio
   # NOTE: multi-modal (image+text+audio) inputs currently only work with Gemini 2.5 Pro or Gemini 2.5 Flash APIs.
   python eval_judgement_audio.py 
   ```

## 5. Citation
Please cite the VRT paper when using this benchmark:

```
@article{vrt2024,
  title   = {Video Reality Test},
  author  = {Author, A. and Researcher, B.},
  journal = {TBD},
  year    = {2024},
  note    = {Update with the official venue once available.}
}
```
