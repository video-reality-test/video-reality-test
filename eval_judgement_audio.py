import os
import json
import torch
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


api_key = "your_api_key_here"
model_name = "gemini-2.5-flash"

PROMPT_DETAILED='''
You are given the frames evenly sampled from an ASMR video. 
<video>
Your task is to determine whether this video is a **real human-recorded ASMR video** or a **synthetically generated one**.

Carefully analyze the frames for:
1. **Visual realism** — look at lighting consistency, texture fidelity, material reflection, and naturalness of motion cues.
2. **Human or object authenticity** — whether skin, hands, tools, or surfaces appear physically plausible rather than rendered or overly smooth.
3. **Acoustic plausibility (implied)** — infer if the scene’s visual rhythm and tactile context align with realistic ASMR sound production (e.g., brushing, tapping, slicing, pouring).
4. **Temporal coherence** — whether transitions between frames follow a physically continuous process or show artifacts of generation (e.g., warping, unstable geometry, or inconsistent textures).

After reasoning, output the answer in the following strict format:

<thinking>
(Write 2–4 sentences summarizing why you believe the video is real or generated.)
<thinking>

<answer>
0
</answer>

where **0** = generated (fake), **1** = real (authentic human-captured video).
'''

PROMPT_DETAILED_reverse='''
You are given the frames evenly sampled from an ASMR video. 
Your task is to determine whether this video is a **real human-recorded ASMR video** or a **synthetically generated one**.

Carefully analyze the frames for:
1. **Visual realism** — look at lighting consistency, texture fidelity, material reflection, and naturalness of motion cues.
2. **Human or object authenticity** — whether skin, hands, tools, or surfaces appear physically plausible rather than rendered or overly smooth.
3. **Acoustic plausibility (implied)** — infer if the scene’s visual rhythm and tactile context align with realistic ASMR sound production (e.g., brushing, tapping, slicing, pouring).
4. **Temporal coherence** — whether transitions between frames follow a physically continuous process or show artifacts of generation (e.g., warping, unstable geometry, or inconsistent textures).

After reasoning, output the answer in the following strict format:

<thinking>
(Write 2–4 sentences summarizing why you believe the video is real or generated.)
<thinking>

<answer>
0
</answer>

where **1** = generated (fake), **0** = real (authentic human-captured video).
<video>
'''

PROMPT_THINK = '''
Given the following video, please determine if the video is real or fake. First, think about the reasoning process, and then provide the answer. The reasoning process should be enclosed within <think> </think> tags, and the answer should be enclosed within <answer> </answer> tags. Respond with "1" for real and "0" for fake. For example: <think> reasoning process here </think><answer> answer here </answer>.
'''

PROMPT_THINK_api = '''
Given the following video, please determine if the video is real or fake. First, think about the reasoning process, and then provide the answer. The reasoning process should be enclosed within <think> </think> tags, and the answer should be enclosed within <answer> </answer> tags. Respond with "1" for real and "0" for fake. For example: <think> reasoning process here </think><answer> answer here </answer>.
<video>
'''

PROMPT_THINK_all = '''
Given the following frames sampled from the video and the audio from the video, please determine if the video is real or fake. First, think about the reasoning process, and then provide the answer. The reasoning process should be enclosed within <think> </think> tags, and the answer should be enclosed within <answer> </answer> tags. Respond with "1" for real and "0" for fake. For example: <think> reasoning process here </think><answer> answer here </answer>.
<video>
'''

PROMPT_NON_THINK = '''
Given the following video, please determine if the video is real or fake. Respond with "1" for real and "0" for fake. Provide the answer directly within <answer> </answer> tags, like this: <answer> answer here </answer>.
'''

PROMPT_NON_THINK_api = '''
Given the following video, please determine if the video is real or fake. Respond with "1" for real and "0" for fake. Provide the answer directly within <answer> </answer> tags, like this: <answer> answer here </answer>.
<video>
'''

PROMPT_NON_THINK_reverse_api = '''
Given the following video, please determine if the video is real or fake. Respond with "0" for real and "1" for fake. Provide the answer directly within <answer> </answer> tags, like this: <answer> answer here </answer>.
<video>
'''


RANDOM_TEST = False


if not RANDOM_TEST:
    pass
else:
    print("Running in random test mode...")


def get_video_path(video_name, data_path):
    if os.path.exists(os.path.join(data_path, video_name)):
        return os.path.join(data_path, video_name)
    return os.path.join(data_path, video_name)


def is_real_video(video_name, data_path):
    if os.path.exists(os.path.join(data_path, video_name)):
        return False
    return True




def inference_one_gpt(video_file, model_name, question):
    import time
    # from openai import OpenAI
    if RANDOM_TEST:
        return '''
        {"semantic_alignment_score": 0.5,
         "justification": "Random test, no ground truth answers provided."}
        '''
    time.sleep(1)
    model_type = model_name
    
    # if test on the visual+audio model
    from videoqa_model import GPT4V
    client = GPT4V(ckpt=api_key, model_stamp=model_type, test_frame=8)
    response = client.qa(video_file, question)
    output = response.strip()
    # print(output)
    return output



def main_proc(save_path, model_name, data_type, data_path):
    prompt_dict = {
        "think": PROMPT_THINK_all,
    }

    video_files = [
        f for f in os.listdir(data_path)
        if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    for prompt_name, question in prompt_dict.items():
        pred_dict = {}
        pred_file = f"{save_path}{data_type}_{prompt_name}.json"

        
        pbar = tqdm(
            enumerate(video_files, 1),
            total=len(video_files),
            desc=f"[{prompt_name}] Processing videos",
            dynamic_ncols=True
        )

        for i, video_file in pbar:
            real_video_path = get_video_path(video_name=video_file, data_path=data_path)
            attempts = 0
            success = False

            while attempts < 3 and not success:
                try:
                    result = inference_one_gpt(real_video_path, model_name, question)
                    pred_dict[video_file] = {"result": result}
                    print(f"Processed {video_file}: {result}")
                    success = True  
                except Exception as e:
                    attempts += 1
                    print(f"Error processing {video_file}: {e}. Retrying {attempts}/3...")

            if not success:
                print(f"Failed to process {video_file} after 3 attempts.")
                continue  

            
            pbar.set_postfix_str(f"Last: {video_file}")

            if i % 10 == 0 or i == len(video_files):
                tmp_file = pred_file + ".tmp"
                with open(tmp_file, "w") as f:
                    json.dump(pred_dict, f, indent=4)
                os.replace(tmp_file, pred_file)

        pbar.close()
        print(f"[{prompt_name}] 已保存到 {pred_file}，共处理 {len(video_files)} 个视频。")



if __name__ == '__main__':
    # video understand by gemini-2.5-flash
    
    # save results path
    save_path_root = f"save/path/root/{model_name}/"
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
        print(f"directory '{save_path_root}' is created.")
    else:
        print(f"directory '{save_path_root}' already exists.")

    print("Evaluating model {} on the judgement dataset...".format(model_name))
    # test data path
    data_path = "/path/to/judgement/dataset/"
    # video generated by veo3.1
    data_type = 'veo3.1'
    print("Evaluating model {} on the judgement dataset...".format(data_type))
    main_proc(save_path_root, model_name, data_type, data_path)
    
