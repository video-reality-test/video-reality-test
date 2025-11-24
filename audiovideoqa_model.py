import openai
from openai import OpenAI
import cv2
import base64
import os
import io
import tempfile
from moviepy.editor import VideoFileClip
import base64

class GPT4V_audio():
    # ... 原有方法保持不变
    def __init__(self, ckpt, model_stamp, test_frame=8):
        self.model_stamp = model_stamp
        if self.model_stamp.startswith('gemini'):
            self.client = OpenAI(api_key=ckpt,
                                base_url="https://api.chataiapi.com/v1")
        elif self.model_stamp.startswith('gpt'):
            self.client = OpenAI(api_key=ckpt,
                                  base_url="https://api.chataiapi.com/v1")
        elif self.model_stamp.startswith('qwen'):
            self.client = OpenAI(api_key=ckpt,
                                  base_url="https://api.chataiapi.com/v1",
                                  )
        else:
            self.client = OpenAI(api_key=ckpt,
                                 base_url= "https://api.probex.top/v1")
        
        # elif isinstance(ckpt, list):
        #     self.client = [OpenAI(api_key=c) for c in ckpt]
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.test_frame = test_frame
        self.resolution = 512

    def _video_to_base64_frames(self, video_path, num_frames=6):
        # ref to https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding
        video = cv2.VideoCapture(video_path)
        base64Frames = []
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(total_frames // num_frames, 1)
        # breakpoint()
        for i in range(total_frames):
            success, frame = video.read()
            if not success:
                break
            if i % frame_interval == 0:
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
                if len(base64Frames) >= num_frames:
                    break
        video.release()
        return base64Frames

    
    def _replace_placeholders(self, prompt: str, images: list, video_len: int, audio_b64: str = None):
        """构建最终消息，包括 text, image, video, audio"""
        result = []
        img_idx = 0

        parts = prompt.split('<video>')
        for i, part in enumerate(parts):
            if i > 0 and img_idx + video_len <= len(images):
                video_urls = [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{images[img_idx + j]}",
                        "size": self.resolution,
                        "detail": "low"
                    }} for j in range(video_len)
                ]
                result.extend(video_urls)
                img_idx += video_len

            image_parts = part.split('<image>')
            for j, text in enumerate(image_parts):
                if j > 0 and img_idx < len(images):
                    image_url = {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{images[img_idx]}",
                        "size": self.resolution,
                        "detail": "low"
                    }}
                    result.append(image_url)
                    img_idx += 1
                if text:
                    result.append({"type": "text", "text": text})

        # 如果有音频，加入 input_audio
        if audio_b64:
            result.append({
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": "wav"  # 也可以支持 mp3
                }
            })

        return result

    
    def _extract_audio_b64_from_video(self, video_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_audio_path = f"{tmpdir}/audio.wav"
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(temp_audio_path, fps=16000, nbytes=2, codec='pcm_s16le', verbose=False, logger=None)
            clip.close()

            with open(temp_audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        return audio_b64
    
    def _get_response(self, client, image: list, audio, prompt, video_len):
        # ref to https://platform.openai.com/docs/guides/vision
        while True:
            try:
                processed_prompt = self._replace_placeholders(prompt, image, video_len, audio_b64=audio)
                # breakpoint()
                if self.model_stamp == 'o1-2024-12-17':
                    response = client.chat.completions.create(
                        model=self.model_stamp,
                        messages=[
                            {
                                "role": "user",
                                "content": processed_prompt
                            }
                        ],
                        # max_completion_tokens=300,  # 300 for text; 2000 for others Not support
                        # temperature=0., Not support
                        seed=42,
                    )
                elif self.model_stamp.startswith('gpt'):
                    response = client.chat.completions.create(
                        model=self.model_stamp,
                        messages=[
                            {
                                "role": "user",
                                "content": processed_prompt
                            }
                        ],
                        max_tokens=2000,  # 300 for text; 2000 for others
                        temperature=0.,
                        seed=42,
                    )
                else: # gemini
                    # models = client.models.list()
                    # for model in models:
                    #     print(model.id)

                    response = client.chat.completions.create(
                        model=self.model_stamp,
                        messages=[
                            {
                                "role": "user",
                                "content": processed_prompt
                            }
                        ],
                        # max_tokens=2000,  # 300 for text; 2000 for others
                        # temperature=0.,
                        # seed=42,
                    )
            except openai.BadRequestError as e:
                if e.code == "sanitizer_server_error":
                    continue
                elif e.code == "content_policy_violation":
                    response = ""
                else:
                    response = f"Error: {e}"
            except openai.InternalServerError as e:
                continue
            # except openai.BadRequestError as e:
            #     response = ""
            break
        return response
    def qa(self, image=None, prompt="", video_desc_flag=True):
        # 处理视频帧和图片
        base64_imgs = []
        v_frames = None
        audio_b64 = None

        if image is not None:
            if image.endswith(".mp4"):
                # 视频抽帧
                v_frames = self._video_to_base64_frames(image, num_frames=self.test_frame)
                base64_imgs += v_frames
                audio_b64 = self._extract_audio_b64_from_video(image)

                # 视频描述
                if video_desc_flag:
                    # prompt = ("The video is split to a series of images sampled "
                    #           "at equal intervals. Answer based on these frames. ") + prompt
                    prompt = ("The video is split to a series of images sampled "
                              "at equal intervals. The audio from the video is here. Answer based on these frames and the audio. ") + prompt

            else:
                with open(image, "rb") as f:
                    base64_imgs.append(base64.b64encode(f.read()).decode("utf-8"))

        # 构建消息，包括图片和音频
        # processed_prompt = self._replace_placeholders(prompt, base64_imgs,
        #                                                video_len=len(v_frames) if v_frames else 0,
        #                                                audio_b64=audio_b64)

        # 发送请求给模型
        response = self._get_response(self.client, base64_imgs, audio_b64, prompt, len(v_frames) if v_frames else 0)
        if isinstance(response, str):
            return response
        else:
            self.completion_tokens += response.usage.completion_tokens
            self.prompt_tokens += response.usage.prompt_tokens
            return response.choices[0].message.content
