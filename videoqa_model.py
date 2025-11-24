import openai
from openai import OpenAI
import cv2
import base64

class GPT4V():
    # model_stamp = 'gpt-4-turbo'

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

    def _replace_placeholders(self, prompt: str, images: list, video_len: int):
        img_idx = 0
        result = []

        # Split the prompt by <video> and <image> to handle each part separately
        parts = prompt.split('<video>')
        for i, part in enumerate(parts):
            if i > 0:  # if this is not the first part, it means we had a <video> placeholder
                if img_idx + video_len <= len(images):
                    # Replace <video> with video_len images
                    video_urls = [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{images[img_idx + j]}",
                                "size": self.resolution,
                                "detail": "low"
                            }
                        }
                        for j in range(video_len)
                    ]
                    result.extend(video_urls)
                    img_idx += video_len

            image_parts = part.split('<image>')
            for j, text in enumerate(image_parts):
                if j > 0:  # if this is not the first sub-part, it means we had an <image> placeholder
                    if img_idx < len(images):
                        image_url = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{images[img_idx]}",
                                "size": self.resolution,
                                "detail": "low"
                            }
                        }
                        result.append(image_url)
                        img_idx += 1

                if text:  # Add the text part
                    result.append({"type": "text", "text": text})

        # breakpoint()
        return result

    def _get_response(self, client, image: list, prompt, video_len):
        # ref to https://platform.openai.com/docs/guides/vision
        while True:
            try:
                processed_prompt = self._replace_placeholders(prompt, image, video_len)
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

    def cost(self):
        # https://openai.com/api/pricing/
        if self.model_stamp == 'gpt-4-turbo':
            return (0.03 * self.completion_tokens + 0.01 * self.prompt_tokens) / 1000  # dollar
        elif self.model_stamp == 'gpt-4o':
            return (0.005 * self.completion_tokens + 0.0015 * self.prompt_tokens) / 1000  # dollar
        elif self.model_stamp == 'gpt-4o-mini':
            return (0.00015 * self.completion_tokens + 0.000075 * self.prompt_tokens) / 1000  # dollar
        elif self.model_stamp == 'o1-2024-12-17':
            return (0.015 * self.completion_tokens + 0.0075 * self.prompt_tokens) / 1000  # dollar
        else:
            raise ValueError(f'not supported {self.model_stamp}')

    def qa(self, image, prompt, video_desc_flag=True):
        # print(self.cost())
        v_frames = None
        if image is not None:
            # to avoid response 'As a text-based AI, I'm unable to view or analyze video content.'
            video_desc = 'The video is split to a series of images sampled at equal intervals from the beginning to the end of it, based on the series of images, answer the question.'
            base64_imgs = []
            # breakpoint()
            # for img in image:
            if image.endswith(".mp4"):
                # Because it may not be enough for self.test_frame, you may need another one to record
                v_frames = self._video_to_base64_frames(image, num_frames=self.test_frame)
                # breakpoint()
                base64_imgs += v_frames
                if video_desc_flag and (video_desc not in prompt):
                    prompt = video_desc + prompt
            else:
                with open(image, "rb") as image_file:
                    base64_imgs.append(base64.b64encode(image_file.read()).decode('utf-8'))
        else:
            base64_imgs = None
        # print(prompt)
        # print(self.client.base_url)
        # print(self.model_stamp)
        if isinstance(self.client, list):
            pointer = 0
            while True:
                client = self.client[pointer]
                try:
                    response = self._get_response(client, base64_imgs, prompt, len(v_frames) if v_frames is not None else None)
                except openai.RateLimitError as e:
                    if pointer < len(self.client) - 1:
                        pointer += 1
                        continue
                    else:
                        raise e
                break
        else:
            print('video frames length in prompt ', len(v_frames))
            response = self._get_response(self.client, base64_imgs, prompt, len(v_frames) if v_frames is not None else None)
        if isinstance(response, str):
            # print(response)  # 使用 print 替换 cprint
            return response
        else:
            self.completion_tokens += response.usage.completion_tokens
            self.prompt_tokens += response.usage.prompt_tokens
            # print(response.choices[0].message.content)  # 使用 print 替换 cprint
            return response.choices[0].message.content


class GPT4O(GPT4V):
    model_stamp = 'gpt-4o'

class GPT4Omini(GPT4V):
    model_stamp = 'gpt-4o-mini'

class O1(GPT4V):
    model_stamp = 'o1-2024-12-17'

class GeminiFlash(GPT4V):
    model_name = 'gemini-2.5-flash'

class GeminiPro(GPT4V):
    model_name = 'gemini-2.5-pro'