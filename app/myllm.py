from utils import Singleton
import utils
import json
from PIL import Image
from lavis.models import load_model_and_preprocess
import llm_api
from typing import List, Dict

import torch
from nudenet import NudeDetector

import logging
logger = logging.getLogger(__name__)

class ImageLLMGen:

    def __init__(self):
        self.titles = {}
        self.captions = {}
        self.lcp = LocalImageCaptioner()
        self.titler = ImageTitler()
        self.remote_llm = RemoteImageLLMGen()

    def get_title(self, image_path, has_nude=True, metadata=None):
        if image_path not in self.titles:
            self._generate(image_path, has_nude, metadata)
        return self.titles.get(image_path, 'Untitled')

    def get_caption(self, image_path, has_nude=True, metadata=None):
        if image_path not in self.captions:
            self._generate(image_path, has_nude, metadata)
        return self.captions.get(image_path, 'No caption')

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

            self.captions[image_path] = caption
            self.titles[image_path] = title
        else:
            logger.debug(f'{image_path} does not contain even mild nudity! processing using remote API')
            title, caption = self.remote_llm.get_llm_gen(image_path, metadata, has_nude = has_nude)

            self.titles[image_path] = title
            self.captions[image_path] = caption


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


def get_photo_info(metadata):
    if metadata is None or not isinstance(metadata, dict):
        return ''

    info_parts = []
    
    # 1. Enhanced Time and Date in Chinese format
    photo_data = metadata.get('photo', {})
    create_date = photo_data.get('create_date', '')
    if create_date:
        try:
            # Parse date and time
            date_part, time_part = create_date.split('T')
            year, month, day = date_part.split('-')
            time_part = time_part.split('+')[0]
            hour, minute, _ = time_part.split(':')
            hour_int = int(hour)
            
            # Format date (Chinese format: 2023年5月15日)
            date_str = f"{year}年{int(month)}月{int(day)}日"
            
            # Determine time period with more detailed descriptions
            if 4 <= hour_int < 6:
                period = "清晨"
            elif 6 <= hour_int < 9:
                period = "早晨"
            elif 9 <= hour_int < 11:
                period = "上午"
            elif 11 <= hour_int < 13:
                period = "中午"
            elif 13 <= hour_int < 17:
                period = "下午"
            elif 17 <= hour_int < 19:
                period = "黄昏"
            else:
                period = "夜晚"
            
            # Format time (12-hour format)
            display_hour = hour_int if hour_int <= 12 else hour_int - 12
            if hour_int == 0:
                display_hour = 12
            
            time_str = f"拍摄时间: {date_str} {period}{display_hour}点{minute}分"
            info_parts.append(time_str)
        except (IndexError, ValueError, AttributeError):
            pass
    
    # 2. Focal length information
    lens_data = metadata.get('lens', {})
    focal_length = lens_data.get('focal_length_mm')
    if focal_length is not None:
        if focal_length > 150:
            focal_str = f"镜头类型: 超长焦拍摄 ({focal_length}mm)"
        elif focal_length > 70:
            focal_str = f"镜头类型: 长焦拍摄 ({focal_length}mm)"
        elif focal_length < 30:
            focal_str = f"镜头类型: 广角拍摄 ({focal_length}mm)"
        else:
            focal_str = f"镜头类型: 标准人眼视角 ({focal_length}mm)"
        info_parts.append(focal_str)
    
    # 3. Lens and camera info
    camera_data = metadata.get('camera', {})
    if camera_data.get('make') and camera_data.get('model'):
        info_parts.append(f"相机: {camera_data['make']} {camera_data['model']}")
    
    if lens_data.get('model'):
        info_parts.append(f"镜头型号: {lens_data['model']}")
    
    # 4. Exposure information
    exposure = photo_data.get('exposure')
    if exposure:
        try:
            numerator, denominator = map(int, exposure.split('/'))
            exposure_value = numerator / denominator
            
            if exposure_value > 1/20:
                info_parts.append("快门类型: 慢门拍摄")
            elif exposure_value < 1/1000:
                info_parts.append("快门类型: 高速快门")
        except (ValueError, ZeroDivisionError):
            pass
    
    # 5. Aperture information
    aperture = lens_data.get('aperture_value')
    if aperture is not None:
        if aperture < 2:
            info_parts.append("光圈类型: 大光圈拍摄")
        elif aperture > 4:
            info_parts.append("光圈类型: 小光圈拍摄")
    
    # 6. GPS information - extract all POIs (up to 10)
    gps_info = []
    gps_resolved = metadata.get('gps_resolved', [])
    
    if isinstance(gps_resolved, list) and len(gps_resolved) > 0:
        # Get base location info from first POI
        base_poi = gps_resolved[0]
        if isinstance(base_poi, dict):
            address = base_poi.get('address', {})
            
            # Build base location string
            location_parts = []
            for key in ['road', 'neighbourhood', 'city', 'state', 'country']:
                if address.get(key):
                    location_parts.append(address[key])
            
            if location_parts:
                gps_info.append(f"基础位置: {'，'.join(location_parts)}")
            
            if base_poi.get('display_name'):
                gps_info.append(f"详细地址: {base_poi['display_name']}")
        
        # Process all POIs (up to 10)
        poi_count = min(10, len(gps_resolved))
        if poi_count > 0:
            gps_info.append("\n附近地点:")
            
            for i in range(poi_count):
                poi = gps_resolved[i]
                if not isinstance(poi, dict):
                    continue
                    
                poi_entry = []
                
                # POI basic info
                poi_name = poi.get('name', '未命名地点')
                poi_entry.append(f"{i+1}. {poi_name}")
                
                # Distance
                if 'distance' in poi:
                    poi_entry.append(f"距离: {poi['distance']}米")
                
                # Type/Class
                poi_type = poi.get('type', '') or poi.get('class', '')
                if poi_type:
                    poi_entry.append(f"类型: {poi_type}")
                
                # Address components
                address = poi.get('address', {})
                address_parts = []
                for key in ['road', 'neighbourhood']:
                    if address.get(key):
                        address_parts.append(address[key])
                
                if address_parts:
                    poi_entry.append(f"位置: {'，'.join(address_parts)}")
                
                gps_info.append(" | ".join(poi_entry))
    
    info_parts.extend(gps_info)
    
    # Format as a nicely indented string with Chinese keys
    result = "\n".join(filter(None, info_parts))
    
    return result
class ImageTitler:
    def __init__(self):
        self.client = llm_api.LLMClient()


    def get_title(self, caption: str, metadata: Dict, lang='zh'):

        assert lang == 'zh', 'only chinese title is supported now'

        metadata_str = get_photo_info(metadata)

        prompt = f"""
        # Task
        Generate a concise, descriptive title (15 Chinese characters or less) for a photo based on:
        
        # Image Caption
        {caption}

        # Other Image Metadata
        {metadata_str}
        
        # Requirement
        Title should:
        - Be poetic yet descriptive
        - Include key elements from caption
        - Reference location if distinctive
        - Consider season if photo date is available, assume Northern Hemisphere unless gps/location says otherwise
        - Consider time of the day (morning / noon / night, etc) if photo time is available
        - Be in Chinese
        - Not exceed 15 chinese characters
        - Avoid generic terms like "photo" or "image"
        - Avoid ，。space, use - & when necessary
        
        Example good titles:
        - 玉渊潭樱花季的午后
        - 故宫角楼落日时分
        - 胡同里的童年
        - 粉樱白樱大光圈特写
        - 清晨目黑川沿岸的慢门
        - 傍晚朝阳公园的长焦人像
        """

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
        metadata_str = get_photo_info(metadata)

        prompt = f"""
        # Task
        1. Generate a concise, descriptive title (15 Chinese characters or less) for a photo based on the provided image and metadata.
        2. Generate a concise, descriptive caption (around 200 english word) for a photo based on the provided image and metadata.

        # Other Image Metadata
        {metadata_str}
        
        # Requirement
        Title should:
        - Be poetic yet descriptive
        - Include key elements from caption
        - Reference location if distinctive
        - Consider season if photo date is available, assume Northern Hemisphere unless gps/location says otherwise
        - Consider time of the day (morning / noon / night, etc) if photo time is available
        - Be in Chinese
        - Not exceed 15 chinese characters
        - Avoid generic terms like "photo" or "image"
        - Avoid ，。space, use - & when necessary

        Caption should 
        - Be detailed about the image and focus on the main part then the other details.
        - Reference location if distinctive
        - Consider season/time if available
        - Be in Chinese
        
        # Example good titles:
        - 玉渊潭樱花季的午后
        - 故宫角楼落日时分
        - 胡同里的童年
        - 粉樱白樱大光圈特写
        - 清晨目黑川沿岸的慢门
        - 傍晚朝阳公园的长焦人像

        # Result format, strictly follow this format and no other should be given.
        title: <the generated title>
        caption: <the generated caption>
        """

        content = self.client.query(
            prompt=prompt,
            image_path=image_path,
            has_nude=has_nude,
            response_format={ 'type': 'text', }
            )

        logger.debug(content)

        title, caption = '', ''
        for l in content.split('\n'):
            kw = 'title:' 
            if not title and kw in l:
                title = l.split(kw)[1].strip()
                continue
            kw = 'caption:' 
            if not caption and kw in l:
                caption = l.split(kw)[1].strip()
                continue
        return title, caption



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

