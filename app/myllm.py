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
        from similarities import utils
        return utils.util.cos_sim(*args, **kw)

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

class ImageTitler:
    def __init__(self):
        self.client = llm_api.LLMClient()

    def _get_photo_info(metadata):
        # 1. Time in Chinese format
        create_date = metadata['photo']['create_date']
        time_part = create_date.split('T')[1].split('+')[0]  # Get time part
        hour, minute, _ = time_part.split(':')
        hour_int = int(hour)
        
        if hour_int < 12:
            time_str = f"时间: 上午{hour_int}点{minute}分"
        else:
            time_str = f"时间: 下午{hour_int-12 if hour_int > 12 else 12}点{minute}分"
        
        # 2. Focal length information
        focal_length = metadata['lens']['focal_length_mm']
        if focal_length > 150:
            focal_str = f"镜头类型: 超长焦拍摄 ({focal_length}mm)"
        elif focal_length > 70:
            focal_str = f"镜头类型: 长焦拍摄 ({focal_length}mm)"
        elif focal_length < 30:
            focal_str = f"镜头类型: 广角拍摄 ({focal_length}mm)"
        else:
            focal_str = f"镜头类型: 标准人眼视角 ({focal_length}mm)"
        
        # 3. Lens and camera info
        camera_str = f"相机: {metadata['camera']['make']} {metadata['camera']['model']}"
        lens_str = f"镜头型号: {metadata['lens']['model']}"
        
        # 4. Exposure information
        exposure = metadata['photo']['exposure']
        numerator, denominator = map(int, exposure.split('/'))
        exposure_value = numerator / denominator
        
        if exposure_value > 1/20:
            exposure_str = "快门类型: 慢门拍摄"
        elif exposure_value < 1/1000:
            exposure_str = "快门类型: 高速快门"
        else:
            exposure_str = ""
        
        # 5. Aperture information
        aperture = metadata['lens']['aperture_value']
        if aperture < 2:
            aperture_str = "光圈类型: 大光圈拍摄"
        elif aperture > 4:
            aperture_str = "光圈类型: 小光圈拍摄"
        else:
            aperture_str = ""
        
        # 6. GPS information (complete dump)
        gps_str = f"GPS数据: {json.dumps(metadata['gps_resolved'], ensure_ascii=False)}"
        
        # Compile all information
        info_parts = [
            time_str,
            focal_str,
            camera_str,
            lens_str,
            exposure_str,
            aperture_str,
            gps_str
        ]
        
        # Format as a nicely indented string with Chinese keys
        result = "\n".join(filter(None, info_parts))
        
        return result

    def get_title(self, caption: str, metadata: Dict, lang='zh'):

        assert lang == 'zh', 'only chinese title is supported now'

        metadata_str = self._get_photo_info(metadata)

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
        - Consider season/time if available
        - Be in Chinese
        - Not exceed 15 chinese characters
        - Avoid generic terms like "photo" or "image"
        
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
                        "longer_title": { "type": "string" },
                        "photography_profession_description": { "type": "string" },
                    },
                    "required": [ "title", "longer_title", "photography_profession_description" ],
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

    def caption(self, img, **kw):
        if self.blip_large_model is None: self._load()
        raw_image = img.convert('RGB')

        # conditional image captioning
        tpl = "a photography of"
        inputs = self.blip_large_processor(raw_image, tpl, return_tensors="pt")

        out = self.blip_large_model.generate(**inputs, **kw)
        res_full = self.blip_large_processor.decode(out[0], skip_special_tokens=True)
        res = res_full[len(tpl):]
        return res.strip()

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



class NudeTagger:

    SENSITIVE_LABELS = {
        # Key: description, threshold (99 means disabled)
        "FEMALE_GENITALIA_COVERED": ('下体隐现',   0.6),
        "FACE_FEMALE":              ('女性面部',   99),
        "BUTTOCKS_EXPOSED":         ('露臀',       0.6),
        "FEMALE_BREAST_EXPOSED":    ('露胸',       0.75),
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
        "FEMALE_BREAST_COVERED":    ('胸隐现',     0.8),
        "BUTTOCKS_COVERED":         ('臀隐现',     0.8),
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

        # mark sensitive label
        marked_res = {
            lbl: {'score': score,
                  'sensitive': is_sensitive(lbl, score),
                  'msg': get_desc(lbl),
                  }
            for lbl, score in res.items()
        }
        return marked_res










# the rest are unused models. please move them out when used
# feel free to remove them when needed

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

