from utils import Singleton
import utils
from PIL import Image
from lavis.models import load_model_and_preprocess

import torch
from nudenet import NudeDetector

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

class ImageCaptioner(Singleton):
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
        "FEMALE_GENITALIA_COVERED":    '下体隐现',
        # "FACE_FEMALE":                '女性面部',
        "BUTTOCKS_EXPOSED":            '露臀',
        "FEMALE_BREAST_EXPOSED":       '露胸',
        "FEMALE_GENITALIA_EXPOSED":    '下体裸露',
        "MALE_BREAST_EXPOSED":         '男露胸',
        "ANUS_EXPOSED":                '肛门裸露',
        # "FEET_EXPOSED":               '脚部裸露',
        # "BELLY_COVERED":              '腹部遮盖',
        # "FEET_COVERED":               '脚部遮盖',
        # "ARMPITS_COVERED":            '腋下遮盖',
        # "ARMPITS_EXPOSED":            '腋下裸露',
        # "FACE_MALE":                  '男性面部',
        # "BELLY_EXPOSED":              '腹部裸露',
        "MALE_GENITALIA_EXPOSED":      '男下体裸露',
        "ANUS_COVERED":                '肛门遮盖',
        "FEMALE_BREAST_COVERED":       '胸隐现',
        "BUTTOCKS_COVERED":            '臀隐现',
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

        # mark sensitive label
        marked_res = {
            lbl: {'score': score, 'sensitive': lbl in self.SENSITIVE_LABELS, 'msg': self.SENSITIVE_LABELS.get(lbl, '')}
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

