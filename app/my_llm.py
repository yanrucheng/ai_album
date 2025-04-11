from utils import Singleton
import utils
import json
from PIL import Image
from lavis.models import load_model_and_preprocess
import llm_api
from typing import List, Dict

import torch
from nudenet import NudeDetector
import my_metadata

import logging
logger = logging.getLogger(__name__)

class ImageLLMGen:

    def __init__(self):
        self.titles = {}
        self.captions = {}
        self.locations = {}
        self.lcp = LocalImageCaptioner()
        self.titler = CaptionTitler()
        self.locator = CaptionLocator()
        self.remote_captioner = RemoteImageCaptioner()

    def get_title(self, image_path, has_nude=True, metadata=None):
        if image_path not in self.titles:
            self._generate_title(image_path, has_nude, metadata)
        return self.titles.get(image_path, 'Untitled')

    def get_caption(self, image_path, has_nude=True, metadata=None):
        if image_path not in self.captions:
            self._generate_caption(image_path, has_nude, metadata)
        return self.captions.get(image_path, 'No caption')

    def get_location(self, image_path, has_nude=True, metadata=None):
        if image_path not in self.locations:
            self._generate_location(image_path, has_nude, metadata)
        return self.locations.get(image_path, None)

    def _generate_caption(self, image_path, has_nude=True, metadata=None):
        if has_nude:
            logger.debug(f'{image_path} contains nudity! processing locally')
            caption = self.lcp.caption(image_path,
                        max_new_tokens=150,     # 限制生成部分长度（确保在100-150范围内）
                        min_new_tokens=50,     # 确保至少生成100个token（可选）
                        num_beams=3,            # 束搜索宽度（提高多样性）
                        # temperature=0.9,        # 适度随机性（避免过于机械）
                        # top_k=50,               # 限制候选词范围（避免低概率词）
                        # top_p=0.95,             # 核采样（保持多样性）
                        repetition_penalty=1.5, # 抑制重复词汇（>1 减少重复）
                        length_penalty=1.2,     # 鼓励稍长输出（>1 增加长度）
                        # do_sample=True,         # 启用采样策略（结合top_k/top_p）
                        early_stopping=True)     # 提前终止（避免冗余）

            self.captions[image_path] = caption
        else:
            logger.debug(f'{image_path} does not contain even mild nudity! processing using remote API')
            caption = self.remote_captioner.get_caption(image_path, metadata, has_nude = has_nude)
            self.captions[image_path] = caption

    def _generate_title(self, image_path, has_nude=True, metadata=None):
        caption = self.get_caption(image_path=image_path, has_nude=has_nude, metadata=metadata)
        location = self.get_location(image_path=image_path, has_nude=has_nude, metadata=metadata)
        title = self.titler.get_title(caption, metadata, location)
        self.titles[image_path] = title

    def _generate_location(self, image_path, has_nude=True, metadata=None):
        caption = self.get_caption(image_path=image_path, has_nude=has_nude, metadata=metadata)
        location = self.locator.get_location(caption, metadata)
        self.locations[image_path] = location

class SimilarityCalculatorABC(Singleton):
    def __init__(self):
        super().__init__()
        self.model = None

    def _load(self):
        pass

    def get_embeddings(self, imgs, **kw):
        if self.model is None: self._load()
        with utils.suppress_c_stdout_stderr(suppress_stderr=True):
            return self.model.get_embeddings(imgs, **kw)

    def similarity_func(self, *args, **kw):
        from similarities import utils as sim_utils
        return sim_utils.util.cos_sim(*args, **kw).squeeze().item()

class ImageSimilarityCalculator(SimilarityCalculatorABC):
    def __init__(self):
        super().__init__()

    def _load(self):
        from similarities import ClipSimilarity
        model_name = 'OFA-Sys/chinese-clip-vit-huge-patch14'
        print('Loading ', model_name)
        self.model = ClipSimilarity(model_name_or_path=model_name)

class TextSimilarityCalculator(SimilarityCalculatorABC):
    def __init__(self):
        super().__init__()

    def _load(self):
        from similarities import BertSimilarity
        model_name = 'shibing624/text2vec-base-chinese'
        print('Loading ', model_name)
        self.model = BertSimilarity(model_name_or_path=model_name)

class Prompt:

    title_system_prompt = '''你是一个专业的图像整理助手，负责根据用户提供的图像信息生成一个不超过12个汉字的中文文件夹名称。名称必须基于可验证的输入数据，确保准确且有意义。

命名规则：
1 包含以下可验证的元素：
   - 地点（来自地标名称或图像描述）
   - 季节（需在图像描述或拍摄日期中有明确证据，12-2月为冬季，3-5月为春季，6-8月为秋季，9-11月为冬季。除非地点/gps明确表示在南半球则以南半球标准为准）
   - 时段（需与元数据中的拍摄时段一致）
   - 视觉特征（需在图像描述中明确提及）

2 禁止包含以下内容：
   - 无法验证的主观描述
   - 缺失的信息
   - 与相邻文件夹重复的或过于相似的名称

处理步骤：
1 检查图像描述中的季节线索
2 核对拍摄日期是否符合季节推断
3 从地标名称提取核心词（如果有）
4 组合可验证的信息，加视觉特征，生成文件夹名称
5 检查是否与相邻文件夹名称冲突

最终输出：仅返回一个不超过12个汉字的中文名称'''

    title_user_prompt = '''图片描述：
{caption}

地点：
{location}

Meta信息：
{meta_info}

此前出现的文件夹名：
{previous_folder_names}'''

    location_system_prompt = '''你的任务是根据图片描述和附近POI列表，严格提取最相关的地点名称
规则：  
1 有效输出：  
   - 必须是具体的地标、场所或景点名称 如"东京塔""明治神宫"  
   - 必须与POI列表完全匹配或在描述中明确提到  
   - 必须以中文输出

2 无效输出：  
   - 单独的国家、城市或区级名称 如"日本""东京"  
   - 地址片段 如"新宿站"若仅为交通枢纽  
   - 无具体名称的泛称 如"公园""博物馆"  

3 优先级：  
   - 优先选择描述中明确提到的地标名称  
   - 若无则从POI列表选择描述最接近且地理位置更接近的匹配
   - 若仍无有效结果 返回不超过区级或市级的名称 如"新宿区""大阪市"  

4 严格限制：  
   - 严禁虚构 必须严格遵循POI列表  
   - 输出仅中文名称 不带标点符号  

输出格式：  
返回具体名称 如"浅草寺" 或以<区/市名称>兜底 如"新宿区" '''

    location_user_prompt = '''图片内容
{caption}

附近POI列表
{geo_candidates}'''


class CaptionLocator:
    def __init__(self, verbosity : utils.Verbosity = utils.Verbosity.Once):
        self.client = llm_api.LLMClient()
        self.verbosity = verbosity

    def get_location(self, caption: str, metadata: Dict):
        geo_candidates = my_metadata.PhotoInfoExtractor(metadata).get_geo_info()
        if geo_candidates == '':
            return None

        user_prompt = Prompt.location_user_prompt.format(caption=caption, geo_candidates=geo_candidates)
        system_prompt = Prompt.location_system_prompt
        if self.verbosity >= utils.Verbosity.Once:
            logger.debug(user_prompt)
        content = self.client.query(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                response_format={
                    'type': 'json_schema',
                    "json_schema": {
                        "name": "photo_locator",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "site_name": { "type": "string" },
                            },
                            "required": [ "site_name" ],
                            "additionalProperties": False
                            }
                        }
                    }
                )
        location = content['site_name']
        if self.verbosity >= utils.Verbosity.Once:
            logger.debug(f'Site: {location}')
        return location

class CaptionTitler:
    def __init__(self, verbosity: utils.Verbosity = utils.Verbosity.Once, max_history: int = 10):
        self.client = llm_api.LLMClient()
        self.verbosity = verbosity
        self.title_deque = collections.deque(maxlen=max_history)  # Now properly initialized with max length
    
    def get_title(self, caption: str, metadata: Dict, location: str):
        metadata_str = my_metadata.PhotoInfoExtractor(metadata).get_info()

        if location is None:
            location = '本图片未记载地理信息，切勿随意假设地点信息。'

        pif = my_metadata.PhotoInfoExtractor(metadata)
        meta_info_parts = [
            pif.get_time_info(),
            pif.get_lens_info(),
            pif.get_camera_info(),
            pif.get_exposure_info(),
            pif.get_aperture_info(),
        ]
        meta_info = '\n'.join(filter(None, meta_info_parts))
        system_prompt = Prompt.title_system_prompt
        
        # Include previous titles in the prompt if available
        previous_titles = '\n'.join(f"- {title}" for title in self.title_deque)
        
        user_prompt = Prompt.title_user_prompt.format(
            caption=caption,
            location=location,
            meta_info=meta_info,
            previous_folder_names=previous_titles if previous_titles else '无历史文件夹名',
        )

        if self.verbosity >= utils.Verbosity.Once:
            logger.debug(user_prompt)

        content = self.client.query(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_format={
                'type': 'json_schema',
                "json_schema": {
                    "name": "photo_naming",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                        },
                        "required": ["title"],
                        "additionalProperties": False
                    }
                }
            }
        )
        title = content['title']
        
        # Store the new title in the history
        self.title_deque.append(title)
        
        if self.verbosity >= utils.Verbosity.Once:
            logger.debug(f'title: {title}')
            if self.title_deque:
                logger.debug(f'Title history (last {len(self.title_deque)}): {list(self.title_deque)}')
                
        return title

    def get_title_history(self) -> List[str]:
        """Returns a list of the last K generated titles"""
        return list(self.title_deque)


class LavisModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model          = None
        self.vis_processors = None
        self.txt_processors = None

    def _get_img(self, img):
        raw_image = img.convert("RGB")
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        return image

    def _get_txt(self, s):
        return self.txt_processors['eval'](s)

class VQA(Singleton, LavisModel):
    def __init__(self):
        super().__init__()

    def _load(self):
        model_name = "blip_vqa"
        print('Loading ', model_name)
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name=model_name, model_type="aokvqa", is_eval=True, device=self.device
        )

    def ask(self, img, question):
        if self.model is None: self._load()

        image = self._get_img(img)
        question = self._get_txt(question)
        samples = {"image": image, "text_input": question}
        return self.model.predict_answers(samples, inference_method="generate")[0]

    def rank(self, img, question, options):
        if self.model is None: self._load()

        image = self._get_img(img)
        question = self._get_txt(question)
        samples = {"image": image, "text_input": question}
        return self.model.predict_answers(samples, answer_list=options, inference_method="rank")[0]


class ImageTextMatcher(Singleton, LavisModel):
    def __init__(self):
        super().__init__()

    def _load(self):
        model_name = 'blip2_image_text_matching'
        print('Loading ', model_name)
        with utils.suppress_c_stdout_stderr(suppress_stderr=True):
            self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name=model_name, model_type='coco', is_eval=True, device=self.device
            )

    def text_match(self, img, txt):
        if self.model is None: self._load()

        with utils.suppress_c_stdout_stderr(suppress_stderr=True):
            image = self._get_img(img)
            txt = self._get_txt(txt)
            itm_output = self.model({"image": image, "text_input": txt}, match_head="itm")
            itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
            return itm_scores[:, 1].item()


class RemoteImageCaptioner:
    def __init__(self, verbosity : utils.Verbosity = utils.Verbosity.Once):
        self.client = llm_api.VLMClient()
        self.verbosity = verbosity

    def get_caption(self, image_path: str, metadata: Dict, has_nude=True):
        assert not has_nude, f'You should not send potentially image with potential nudity ({image_path}) remotely.\n'

        system_prompt = '''你是一个专业的图像描述生成模型，专注于准确识别图像中的地点信息和人物特征。请遵循以下规则：
1. 地点描述：
   - 仅当图像中明确显示地标、文字标识、独特建筑或明确地理特征时，才描述具体位置
   - 避免推测季节或时间（除非有钟表/昼夜明显特征）
   - 区分室内/室外场景，注意墙面材质、地面类型、门窗样式等建筑细节
   - 记录文字信息（招牌、路牌等）需严格准确

2. 人物描述：
   - 记录可见的性别、年龄范围、衣着颜色/款式
   - 注明显著特征（眼镜、纹身、配饰等）
   - 描述动作和互动关系（如有多个人物）
   - 绝对避免种族、职业等主观推测

3. 输出格式要求：
   - 使用中性客观的语言
   - 地点和人物信息分段落描述
   - 不确定的元素用"可能"、"似乎"等谨慎表述'''

        user_prompt = '''请分析该图像并生成详细描述，特别注意：
1. 地点信息：
   - [图像中明确可见的地标或文字线索是什么？]
   - [建筑风格/自然环境有哪些可验证的特征？]
   - [地面/墙面材质等细节是否有助于定位？]

2. 人物信息：
   - [可见的人物数量及基本特征？]
   - [服装的显著颜色和款式？]
   - [是否有携带物品或进行特定动作？]

请严格区分观察事实与推测，对模糊信息保持谨慎。描述请用中文输出500字左右。'''

        if self.verbosity >= utils.Verbosity.Once:
            logger.debug('Image Captioning:', user_prompt)

        content = self.client.query(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image_path=image_path,
            has_nude=has_nude,
            response_format={ 'type': 'text' }
            )

        if self.verbosity >= utils.Verbosity.Once:
            logger.debug(f'Caption:', content)

        return content



class LocalImageCaptioner(Singleton):
    def __init__(self):
        super().__init__()
        self.blip_large_model = None

    def _load(self):
        from transformers import BlipProcessor, BlipForConditionalGeneration

        model_name = "Salesforce/blip-image-captioning-large"
        print('Loading ', model_name)
        self.blip_large_processor = BlipProcessor.from_pretrained(model_name)
        self.blip_large_model = BlipForConditionalGeneration.from_pretrained(model_name)

    def caption(self, image_path, **kw):
        if self.blip_large_model is None: self._load()

        raw_image = None
        with Image.open(image_path) as img:
            img.load()
            raw_image = img.convert('RGB')

        # conditional image captioning
        tpl = "a photography of"
        inputs = self.blip_large_processor(raw_image, tpl, return_tensors="pt")

        out = self.blip_large_model.generate(**inputs, **kw)
        res_full = self.blip_large_processor.decode(out[0], skip_special_tokens=True)
        res = res_full[len(tpl):]
        return res.strip()


class NudeTagger:

    SENSITIVE_LABELS = {
        # Key: description, threshold (99 means disabled)
        "FEMALE_GENITALIA_COVERED": ('下体隐现',   0.5),
        "FACE_FEMALE":              ('女性面部',   99),
        "BUTTOCKS_EXPOSED":         ('露臀',       0.5),
        "FEMALE_BREAST_EXPOSED":    ('露胸',       0.5),
        "FEMALE_GENITALIA_EXPOSED": ('下体裸露',   0.4),
        "MALE_BREAST_EXPOSED":      ('男露胸',     99),
        "ANUS_EXPOSED":             ('肛门裸露',   0.5),
        "FEET_EXPOSED":             ('脚部裸露',   99),
        "BELLY_COVERED":            ('腹部遮盖',   99),
        "FEET_COVERED":             ('脚部遮盖',   99),
        "ARMPITS_COVERED":          ('腋下遮盖',   99),
        "ARMPITS_EXPOSED":          ('腋下裸露',   99),
        "FACE_MALE":                ('男性面部',   99),
        "BELLY_EXPOSED":            ('腹部裸露',   99),
        "MALE_GENITALIA_EXPOSED":   ('男下体裸露', 0.6),
        "ANUS_COVERED":             ('肛门遮盖',   0.5),
        "FEMALE_BREAST_COVERED":    ('胸隐现',     0.5),
        "BUTTOCKS_COVERED":         ('臀隐现',     0.5),
    }

    def __init__(self):
        self.nude_detector = NudeDetector()

    def detect(self, img_path):
        l = self.nude_detector.detect(img_path)
        res = {}
        for d in l:
            lbl, score = d['class'], d['score']
            if lbl not in res:
                res[lbl] = score
            res[lbl] = max(res[lbl], score)

        assert all(lbl in self.SENSITIVE_LABELS for lbl in res), \
                f"This should not happen. Unrecognized nude tag. got: {', '.join(res.keys())}"

        def get_desc(lbl):
            return self.SENSITIVE_LABELS.get(lbl, ('',))[0]

        def is_sensitive(lbl, score):
            return score >= self.SENSITIVE_LABELS.get(lbl, ('',99))[1]

        def is_mild_sensitive(lbl, score):
            s = self.SENSITIVE_LABELS.get(lbl, ('',99))[1] / 3
            return score >= (self.SENSITIVE_LABELS.get(lbl, ('',99))[1] / 3)

        # mark sensitive label
        marked_res = {
            lbl: {'score': score,
                  'sensitive': is_sensitive(lbl, score),
                  'mild_sensitive': is_mild_sensitive(lbl, score),
                  'msg': get_desc(lbl),
                  }
            for lbl, score in res.items()
        }
        return marked_res










# the rest are unused models. please move them out when used
# feel free to remove them when needed

class ImageCaptionerChinese(Singleton):
    def __init__(self):
        super().__init__()
        self.captioner = ImageCaptioner()
        self.translator = MyTranslator()

    def caption(self, img, **kw):
        res_eng = self.captioner.caption(img, **kw)
        res_ch = self.translator.translate(res_eng)
        return res_ch

class MyTranslator(Singleton):
    def __init__(self):
        super().__init__()
        self.model = None

    def _load(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

    def translate(self, s_eng, max_length=15):
        if self.model is None: self._load()

        # Translate text
        text_to_translate = s_eng
        inputs = self.tokenizer.encode(text_to_translate, return_tensors="pt")
        outputs = self.model.generate(inputs, max_new_tokens=max_length)
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return translated_text



class ImageCaptionerBlipLargeCOCO_unused(Singleton, LavisModel):
    def __init__(self):
        super().__init__()

    def _load(self):
        model_name = 'blip_caption'
        print('Loading ', model_name)
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name=model_name, model_type='large_coco', is_eval=True, device=self.device
        )

    def caption(self, img, **kw):
        if self.model is None: self._load()

        image = self._get_img(img)
        return self.model.generate({"image": image}, **kw)[0]


class VQA_uform_unused(Singleton):
    def __init__(self):
        super().__init__()

    def _load(self):
        from uform.gen_model import VLMForCausalLM, VLMProcessor

        model_name = "unum-cloud/uform-gen-chat"
        print('Loading ', model_name)
        self.model = VLMForCausalLM.from_pretrained(mode_name)
        self.processor = VLMProcessor.from_pretrained(model_name)

    def ask(self, img, question="What do you see?", **kw):
        if self.model is None: self._load()

        inputs = processor(texts=[question], images=[img], return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=128,
                eos_token_id=32001,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        prompt_len = inputs["input_ids"].shape[1]
        decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
        return decoded_text

