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
        self.titler = ImageTitler()
        self.locator = ImageLocator()
        self.remote_llm = RemoteImageLLMGen()

    def get_title(self, image_path, has_nude=True, metadata=None):
        if image_path not in self.titles:
            self._generate(image_path, has_nude, metadata)
        return self.titles.get(image_path, 'Untitled')

    def get_caption(self, image_path, has_nude=True, metadata=None):
        if image_path not in self.captions:
            self._generate(image_path, has_nude, metadata)
        return self.captions.get(image_path, 'No caption')

    def get_location(self, image_path, has_nude=True, metadata=None):
        if image_path not in self.locations:
            self._generate(image_path, has_nude, metadata)
        return self.locations.get(image_path, 'Unknown')

    def _generate(self, image_path, has_nude=True, metadata=None):
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
            title = self.titler.get_title(caption, metadata)
            geo_candidates = my_metadata.PhotoInfoExtractor(metadata).get_geo_info()
            location = self.locator.get_location(caption, geo_candidates)

            self.captions[image_path] = caption
            self.titles[image_path] = title
            self.locations[image_path] = location
        else:
            logger.debug(f'{image_path} does not contain even mild nudity! processing using remote API')
            title, caption, location = self.remote_llm.get_llm_gen(image_path, metadata, has_nude = has_nude)

            self.titles[image_path] = title
            self.captions[image_path] = caption
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

    caption_requirement  = '''Caption should 
- Be detailed about the image and focus on the main part then the other details.
- Reference location if distinctive
- Consider season if photo date is available, assume Northern Hemisphere unless gps/location says otherwise
- Consider time of the day (morning / noon / night, etc) if photo time is available
- Be in Chinese
- Only suggest dark or morbid titles if the photography themes explicitly mentions themes like
    cemeteries, funerals, or horror. Otherwise, assume the photo is neutral/positive and avoid such terms entirely.
    Prioritize safe, generic, or uplifting titles by default.
- Sometimes photos are taken near cemeteries but no explicit signs can be identified on the photo. Do not directly assume the theme to be dark / morbid in this case.'''

    title_task = 'Generate a concise, descriptive title (15 Chinese characters or less) for a photo based on the provided image and metadata.'

    title_requirement = '''Title should:
- Be poetic yet descriptive
- Include key elements from caption
- Reference location if distinctive
- Consider season if photo date is available, assume Northern Hemisphere unless gps/location says otherwise
- Consider time of the day (morning / noon / night, etc) if photo time is available
- Be in Chinese
- Not exceed 15 chinese characters
- Avoid generic terms like "photo" or "image"
- Avoid ，。space, use - & when necessary
- Only suggest dark or morbid titles if the photography themes explicitly mentions themes like
    cemeteries, funerals, or horror. Otherwise, assume the photo is neutral/positive and avoid such terms entirely.
    Prioritize safe, generic, or uplifting titles by default.
- Sometimes photos are taken near cemeteries but no explicit signs can be identified on the photo. Do not directly assume the theme to be dark / morbid in this case.'''

    location_task = '''Summarize the most likely location among the Point of interests, based on the image content. NEVER directly give me the full address to me for this field. If no Geo info provided, return Unknown for this field. '''

    location_requirement = '''Location should:
- Be within 10 Chinese characters
- Be detailed. The following 2 examples are for your reference only:
    1. <specific attraction within the garden> better than <garden name> and better than <country and city name>
    2. <shop name with 地标> better than <a shop> better than <city name> only
- Only provide city and country if you cannot find any more detailed clue. Otherwise ignore the city and country name.
- Avoid punctuation like ，。and spacing
- A point of interest with shorter distance is more likely to be better, but do consider the image/image-caption content. the best location should fit the image.'''

class ImageLocator:
    def __init__(self):
        self.client = llm_api.LLMClient()


    def get_location(self, caption: str, geo_candidates: str):

        prompt = f"""# Task
{Prompt.location_task}

# Image Caption
{caption}

# Point of interests
{geo_candidates}

# Requirement
{Prompt.location_requirement}"""

        logger.debug(prompt)
        content = self.client.query(prompt, response_format={
            'type': 'json_schema',
            "json_schema": {
                "name": "photo_locator",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "location": { "type": "string" },
                    },
                    "required": [ "location" ],
                    "additionalProperties": False
                    }
                }
            })
        return content['location']

class ImageTitler:
    def __init__(self):
        self.client = llm_api.LLMClient()


    def get_title(self, caption: str, metadata: Dict, lang='zh'):

        assert lang == 'zh', 'only chinese title is supported now'

        metadata_str = my_metadata.PhotoInfoExtractor(metadata).get_info()

        prompt = f"""# Task
{Prompt.title_task}

# Image Caption
{caption}

# Other Image Metadata
{metadata_str}

# Requirement
{Prompt.title_requirement} """

        logger.debug(prompt)
        content = self.client.query(prompt, response_format={
            'type': 'json_schema',
            "json_schema": {
                "name": "photo_naming",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": { "type": "string" },
                    },
                    "required": [ "title" ],
                    "additionalProperties": False
                    }
                }
            })
        return content['title']


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


class RemoteImageLLMGen:
    def __init__(self):
        self.client = llm_api.VLMClient()

    def get_llm_gen(self, image_path: str, metadata: Dict, has_nude=True):
        metadata_str = my_metadata.PhotoInfoExtractor(metadata).get_info()

        prompt = f"""# Task
1. {Prompt.title_task}
2. {Prompt.location_task}
3. Generate a concise, descriptive caption (300 Chinese characters or less) for a photo based on the provided image and metadata. Also concisely mention your inferring reason for the location here.

# Other Image Metadata including location candidates
{metadata_str}

# Caption Requirement
{Prompt.caption_requirement}

# Title Requirement
{Prompt.title_requirement}

# Location Requirement
{Prompt.location_requirement}

# Result format, strictly follow this format and no other should be given.
title: <the generated title, a must field>
location: <the infered location, an must field>
caption: <the generated caption, a must field> """

        logger.debug(prompt)
        content = self.client.query(
            prompt=prompt,
            image_path=image_path,
            has_nude=has_nude,
            response_format={ 'type': 'text', }
            )

        logger.debug(content)

        title = caption = location = ''
        for l in content.split('\n'):
            kw = 'title:' 
            if not title and kw in l:
                title = l.split(kw)[1].strip()
                continue
            kw = 'caption:' 
            if not caption and kw in l:
                caption = l.split(kw)[1].strip()
                continue
            kw = 'location:' 
            if not location and kw in l:
                location = l.split(kw)[1].strip()
                continue
        location = location or 'Unknown'
        return title, caption, location



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

