from similarities import ClipSimilarity, utils
from utils import Singleton
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch
from nudenet import NudeDetector

class ImageSimilarityCalculator(Singleton):
    def __init__(self):
        super().__init__()
        self.model = None
        
    def _load(self):
        model_name = 'OFA-Sys/chinese-clip-vit-huge-patch14'
        print('Loading ', model_name)
        self.model = ClipSimilarity(model_name_or_path=model_name)

    def get_embeddings(self, imgs, **kw):
        if self.model is None: self._load()
        return self.model.get_embeddings(imgs, **kw)

    def similarity_func(self, *args, **kw):
        return utils.util.cos_sim(*args, **kw)


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
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name=model_name, model_type='coco', is_eval=True, device=self.device
        )

    def text_match(self, img, txt):
        if self.model is None: self._load()
            
        image = self._get_img(img)
        txt = self._get_txt(txt)
        itm_output = self.model({"image": image, "text_input": txt}, match_head="itm")
        itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
        return itm_scores[:, 1].item()

class ImageCaptioner(Singleton, LavisModel):
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
        return self.model.generate({"image": image}, **kw)

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
            









