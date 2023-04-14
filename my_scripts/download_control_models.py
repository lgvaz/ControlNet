# 2.1. Download ControlNet Model 
import os

installv2Models = []

#@markdown ### Available Model
#@markdown You must download the model before launching the Gradio interface
#@markdown Select one of available model to download:

repo_dir = "$HOME/git/ControlNet"


modelUrl = ["", \
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth", \
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth", \
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_hed.pth", \
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_mlsd.pth", \
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_normal.pth", \
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth", \
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth", \
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_seg.pth"]
modelList = ["", \
             "control_sd15_canny", \
             "control_sd15_depth", \
             "control_sd15_hed", \
             "control_sd15_mlsd", \
             "control_sd15_normal", \
             "control_sd15_openpose", \
             "control_sd15_scribble", \
             "control_sd15_seg" ]
modelName = "control_sd15_canny" #@param ["control_sd15_canny", "control_sd15_depth", "control_sd15_hed", "control_sd15_mlsd", "control_sd15_normal", "control_sd15_openpose", "control_sd15_scribble", "control_sd15_seg"]

installModels = [(modelName, modelUrl[modelList.index(modelName)]) for modelName in modelList]

def install(checkpoint_name, url):
  ext = "pth" if url.endswith(".pth") else "pt"

  hf_token = '' 
  user_header = f"\"Authorization: Bearer {hf_token}\""
  os.system(f"aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {repo_dir}/models -o {checkpoint_name}.{ext} '{url}'")

def install_checkpoint():
  for model in installModels:
    install(model[0], model[1])

install_checkpoint()
